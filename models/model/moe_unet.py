"""MoE-UNet: UNet with Mixture-of-Experts FFN for KineRoute paired sparse routing.

Each BasicTransformerBlock's FeedForward is replaced with an MoE-FFN bank.
A shared paired router dispatches exactly one (joint expert, pose expert) per layer.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig

from models.model.utils import (
    FeedForward,
    CrossAttention,
    Normalize,
    timestep_embedding,
    default,
)
from models.model.scene_model import create_scene_model
from models.base import MODEL


# ---------------------------------------------------------------------------
# MoE FeedForward Bank
# ---------------------------------------------------------------------------

class MoEFeedForwardBank(nn.Module):
    """A bank of `num_experts` FeedForward networks.

    At inference only one expert is evaluated (selected externally).
    During training with Gumbel-STE, a hard one-hot selects one expert
    but gradients flow through the Gumbel-Softmax relaxation.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        mult: int = 2,
        glu: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [FeedForward(dim, mult=mult, glu=glu, dropout=dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Run a single expert on the full input."""
        return self.experts[expert_idx](x)

    def forward_soft(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Weighted combination of all experts (used during Gumbel-STE training).

        Args:
            x: [B, L, D]
            weights: [num_experts] soft weights (one-hot via STE)
        """
        out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            if weights[i].item() > 1e-6 or self.training:
                out = out + weights[i] * expert(x)
        return out


# ---------------------------------------------------------------------------
# Paired Router — the core KineRoute mechanism
# ---------------------------------------------------------------------------

class PairedRouter(nn.Module):
    """FK-supervised Gumbel-Softmax paired router.

    For each layer, produces a joint distribution over (joint_expert, pose_expert)
    pairs and selects exactly one pair via Gumbel-STE (train) or argmax (eval).

    Input context:
        r_t^l = MLP([sigma_t; z_scene; Pool(h_joint^l); Pool(h_pose^l)])

    Score matrix:
        score_ij = u_i @ r + v_j @ r + B_ij

    Selection:
        train: Gumbel-Softmax STE → hard one-hot
        eval:  argmax
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        context_dim: int = 512,
        tau_init: float = 1.0,
        tau_min: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.tau = tau_init
        self.tau_min = tau_min

        # Context builder: [sigma_embed; z_scene; pool_joint; pool_pose] -> r
        # sigma_embed: d_model, z_scene: context_dim, pool_joint: d_model, pool_pose: d_model
        ctx_input_dim = d_model + context_dim + d_model + d_model
        self.context_mlp = nn.Sequential(
            nn.Linear(ctx_input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # Score decomposition: u_i for joint experts, v_j for pose experts
        self.W_u = nn.Linear(d_model, num_experts, bias=False)  # joint expert scores
        self.W_v = nn.Linear(d_model, num_experts, bias=False)  # pose expert scores

        # Pair interaction bias B_ij: learnable [num_experts, num_experts]
        self.B_ij = nn.Parameter(torch.zeros(num_experts, num_experts))

    def set_tau(self, tau: float):
        self.tau = max(tau, self.tau_min)

    def forward(
        self,
        sigma_embed: torch.Tensor,
        z_scene: torch.Tensor,
        h_joint_pool: torch.Tensor,
        h_pose_pool: torch.Tensor,
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
        """Compute paired routing decision.

        Args:
            sigma_embed: [B, d_model] — timestep embedding
            z_scene: [B, context_dim] — scene latent (mean-pooled)
            h_joint_pool: [B, d_model] — mean-pooled joint branch hidden states
            h_pose_pool: [B, d_model] — mean-pooled pose branch hidden states

        Returns:
            joint_idx: selected joint expert index
            pose_idx: selected pose expert index
            pair_probs: [num_experts, num_experts] probability matrix
            pair_logits: [num_experts, num_experts] raw logits
        """
        # Build context
        ctx_input = torch.cat([sigma_embed, z_scene, h_joint_pool, h_pose_pool], dim=-1)
        # Average over batch to get a single routing decision
        ctx_input = ctx_input.mean(dim=0, keepdim=True)  # [1, ctx_dim]
        r = self.context_mlp(ctx_input)  # [1, d_model]

        # Compute score matrix
        u = self.W_u(r).squeeze(0)  # [num_experts]
        v = self.W_v(r).squeeze(0)  # [num_experts]

        # score_ij = u_i + v_j + B_ij
        score_matrix = u.unsqueeze(1) + v.unsqueeze(0) + self.B_ij  # [E, E]
        logits_flat = score_matrix.reshape(-1)  # [E*E]

        if self.training:
            # Gumbel-Softmax STE
            pair_onehot = F.gumbel_softmax(logits_flat, tau=self.tau, hard=True)
        else:
            # Deterministic argmax
            idx = logits_flat.argmax()
            pair_onehot = torch.zeros_like(logits_flat)
            pair_onehot[idx] = 1.0

        pair_matrix = pair_onehot.reshape(self.num_experts, self.num_experts)
        pair_probs = F.softmax(logits_flat, dim=0).reshape(self.num_experts, self.num_experts)

        # Extract selected indices
        flat_idx = pair_onehot.argmax().item()
        joint_idx = flat_idx // self.num_experts
        pose_idx = flat_idx % self.num_experts

        return joint_idx, pose_idx, pair_probs, score_matrix, pair_matrix


# ---------------------------------------------------------------------------
# MoE BasicTransformerBlock
# ---------------------------------------------------------------------------

class MoEBasicTransformerBlock(nn.Module):
    """BasicTransformerBlock with MoE-FFN instead of dense FFN.

    The FFN is an MoEFeedForwardBank; the expert is selected externally.
    Attention layers are kept frozen (identical to dense CoKin).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        gated_ff: bool = True,
        mult_ff: int = 2,
        num_experts: int = 4,
    ):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)
        self.moe_ff = MoEFeedForwardBank(dim, num_experts=num_experts, mult=mult_ff, glu=gated_ff, dropout=dropout)
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, expert_idx: int = 0) -> torch.Tensor:
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.moe_ff(self.norm3(x), expert_idx=expert_idx) + x
        return x

    def forward_with_dense_output(
        self, x: torch.Tensor, dense_ff: FeedForward, context: Optional[torch.Tensor] = None, expert_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass that also returns the dense FFN output for KD loss.

        Returns:
            x: output after MoE FFN
            moe_ffn_out: raw MoE FFN output (before residual)
            dense_ffn_out: raw dense FFN output (for distillation)
        """
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        normed = self.norm3(x)
        moe_ffn_out = self.moe_ff(normed, expert_idx=expert_idx)
        with torch.no_grad():
            dense_ffn_out = dense_ff(normed)
        x = moe_ffn_out + x
        return x, moe_ffn_out, dense_ffn_out


# ---------------------------------------------------------------------------
# MoE SpatialTransformer
# ---------------------------------------------------------------------------

class MoESpatialTransformer(nn.Module):
    """SpatialTransformer with MoE-FFN blocks."""

    def __init__(
        self,
        in_channels: int,
        n_heads: int = 8,
        d_head: int = 64,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        mult_ff: int = 2,
        num_experts: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                MoEBasicTransformerBlock(
                    inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, mult_ff=mult_ff,
                    num_experts=num_experts,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, expert_idx: int = 0) -> torch.Tensor:
        B, C, L = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c l -> b l c")
        for block in self.transformer_blocks:
            x = block(x, context=context, expert_idx=expert_idx)
        x = rearrange(x, "b l c -> b c l")
        x = self.proj_out(x)
        return x + x_in


# ---------------------------------------------------------------------------
# MoE UNetModel
# ---------------------------------------------------------------------------

@MODEL.register()
class MoEUNetModel(nn.Module):
    """UNet with MoE-FFN layers for KineRoute.

    Architecture is identical to UNetModel except the SpatialTransformer blocks
    use MoE-FFN banks instead of dense FFN. The expert selection is provided
    externally via the `expert_idx` argument.
    """

    def __init__(self, cfg: DictConfig, slurm: bool = False, *args, **kwargs) -> None:
        super().__init__()

        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding
        self.num_experts = int(cfg.get("num_experts", 4))

        # Scene model
        self.scene_model_name = cfg.scene_model.name
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == "PointNet":
            scene_model_args = {
                "c": scene_model_in_dim,
                "num_points": cfg.scene_model.num_points,
                "num_tokens": cfg.scene_model.num_tokens,
            }
        else:
            scene_model_args = {"c": scene_model_in_dim, "num_points": cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)

        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.in_layers = nn.Sequential(nn.Conv1d(self.d_x, self.d_model, 1))

        from models.model.utils import ResBlock

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(self.d_model, time_embed_dim, self.resblock_dropout, self.d_model)
            )
            self.layers.append(
                MoESpatialTransformer(
                    self.d_model,
                    self.transformer_num_heads,
                    self.transformer_dim_head,
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                    num_experts=self.num_experts,
                )
            )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        cond: torch.Tensor,
        expert_indices: Optional[list] = None,
    ) -> torch.Tensor:
        """Forward pass with MoE expert selection.

        Args:
            x_t: [B, L, C] or [B, C]
            ts: [B] timesteps
            cond: [B, N, C_cond] scene features
            expert_indices: list of int, one per block. If None, uses expert 0.
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)
        h = rearrange(x_t, "b l c -> b c l")
        h = self.in_layers(h)

        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX)
            h = h + pos_embedding_Q.permute(1, 0)

        if expert_indices is None:
            expert_indices = [0] * self.nblocks

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)  # ResBlock
            h = self.layers[i * 2 + 1](h, context=cond, expert_idx=expert_indices[i])  # MoE SpatialTransformer

        h = self.out_layers(h)
        h = rearrange(h, "b c l -> b l c")

        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def get_intermediate_features(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Run through layers up to (but not including) the MoE-FFN and return
        the intermediate hidden state after each block's ResBlock + Attention.
        Used by the router to get h_joint and h_pose for context building.

        Returns mean-pooled hidden state: [B, d_model]
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)

        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)
        h = rearrange(x_t, "b l c -> b c l")
        h = self.in_layers(h)

        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX)
            h = h + pos_embedding_Q.permute(1, 0)

        # Return mean-pooled hidden after in_layers (before any transformer block)
        return h.mean(dim=-1)  # [B, d_model]

    def condition(self, data: Dict) -> torch.Tensor:
        """Obtain scene feature with scene model (same as UNetModel)."""
        if self.scene_model_name == "PointTransformer":
            b = data["offset"].shape[0]
            pos, feat, offset = data["pos"], data["feat"], data["offset"]
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, "(b n) c -> b n c", b=b, n=self.scene_model.num_groups)
        elif self.scene_model_name == "PointNet":
            b = data["pos"].shape[0]
            pos = data["pos"].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        else:
            raise Exception("Unexpected scene model.")
        return scene_feat
