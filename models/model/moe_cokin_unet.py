from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from models.base import MODEL
from models.m2diffuser.moe_cokin_core import (
    MoE_CoKin_TransformerBlock,
    RouterOutput,
)
from models.model.scene_model import create_scene_model
from models.model.utils import timestep_embedding


@MODEL.register()
class MoECoKinUNetModel(nn.Module):
    """MoE-CoKin denoiser with noise-conditioned routing blocks."""

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super().__init__()
        self.d_x = int(cfg.d_x)
        self.d_model = int(cfg.d_model)
        self.nblocks = int(cfg.nblocks)
        self.context_dim = int(cfg.context_dim)
        self.use_position_embedding = bool(cfg.use_position_embedding)

        self.branch = str(cfg.get("branch", "joint"))
        if self.branch not in {"pose", "joint"}:
            raise ValueError(f"Unsupported branch type: {self.branch}")

        self.transformer_num_heads = int(cfg.transformer_num_heads)
        self.transformer_dropout = float(cfg.transformer_dropout)
        self.noise_emb_dim = int(cfg.get("noise_emb_dim", self.d_model * 2))
        self.num_experts = int(cfg.get("num_experts", 8))
        self.top_k = int(cfg.get("top_k", 2))
        self.expert_hidden_dim = int(cfg.get("expert_hidden_dim", self.d_model * 4))
        self.moe_dropout = float(cfg.get("moe_dropout", self.transformer_dropout))

        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = (
            3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        )
        if cfg.scene_model.name == "PointNet":
            scene_model_args = {
                "c": scene_model_in_dim,
                "num_points": cfg.scene_model.num_points,
                "num_tokens": cfg.scene_model.num_tokens,
            }
        else:
            scene_model_args = {
                "c": scene_model_in_dim,
                "num_points": cfg.scene_model.num_points,
            }
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)

        weight_path = (
            cfg.scene_model.pretrained_weights_slurm
            if slurm
            else cfg.scene_model.pretrained_weights
        )
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for param in self.scene_model.parameters():
                param.requires_grad_(False)

        self.scene_proj: nn.Module
        if self.context_dim == self.d_model:
            self.scene_proj = nn.Identity()
        else:
            self.scene_proj = nn.Linear(self.context_dim, self.d_model)

        self.in_layers = nn.Sequential(nn.Conv1d(self.d_x, self.d_model, 1))
        self.blocks = nn.ModuleList(
            [
                MoE_CoKin_TransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.transformer_num_heads,
                    noise_emb_dim=self.noise_emb_dim,
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    expert_hidden_dim=self.expert_hidden_dim,
                    dropout=self.moe_dropout,
                )
                for _ in range(self.nblocks)
            ]
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )

        self.latest_route_outputs: List[RouterOutput] = []
        self.latest_router_output: Optional[RouterOutput] = None

    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the model to noisy trajectory with scene condition."""
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, L, C], got shape={tuple(x_t.shape)}")

        h = rearrange(x_t, "b l c -> b c l")
        h = self.in_layers(h)
        h = rearrange(h, "b c l -> b l c")

        if self.use_position_embedding:
            pos = torch.arange(h.shape[1], dtype=torch.float32, device=h.device)
            pos_emb = timestep_embedding(pos, self.d_model).to(dtype=h.dtype)
            h = h + pos_emb.unsqueeze(0)

        if cond.ndim != 3:
            raise ValueError(f"cond must be [B, N, C], got shape={tuple(cond.shape)}")
        scene_tokens = self.scene_proj(cond)

        routes: List[RouterOutput] = []
        for block in self.blocks:
            h, route = block(
                x=h,
                scene_tokens=scene_tokens,
                t=ts,
                branch=self.branch,
            )
            routes.append(route)

        self.latest_route_outputs = routes
        self.latest_router_output = routes[-1] if len(routes) > 0 else None

        h = rearrange(h, "b l c -> b c l")
        h = self.out_layers(h)
        h = rearrange(h, "b c l -> b l c")

        if in_shape == 2:
            h = h.squeeze(1)
        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """Extract scene condition tokens from input batch."""
        if self.scene_model_name == "PointTransformer":
            batch_size = data["offset"].shape[0]
            pos, feat, offset = data["pos"], data["feat"], data["offset"]
            _, x5, _ = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(
                x5,
                "(b n) c -> b n c",
                b=batch_size,
                n=self.scene_model.num_groups,
            )
        elif self.scene_model_name == "PointNet":
            batch_size = data["pos"].shape[0]
            pos = data["pos"].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(
                batch_size, self.scene_model.num_groups, -1
            )
        else:
            raise ValueError(f"Unexpected scene model: {self.scene_model_name}")

        return scene_feat
