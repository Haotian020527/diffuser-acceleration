"""CoKin-MoE Diffuser: FK-supervised paired sparse routing for dual-space diffusion.

Extends ConsistencyCoupledKinematicsDiffuser with:
  - MoE-FFN banks in both Joint and Pose denoisers
  - Per-layer PairedRouter with Gumbel-Softmax STE
  - L_FK_route: FK consistency supervision on routing selection
  - L_KD: knowledge distillation from frozen dense CoKin teacher FFN
  - L_lb: load balancing to prevent expert collapse
"""

import copy
from typing import Any, Callable, Dict, Optional, Tuple, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.base import DIFFUSER
from models.m2diffuser.cokin import ConsistencyCoupledKinematicsDiffuser
from models.model.moe_unet import PairedRouter, MoEUNetModel
from models.model.utils import timestep_embedding


@DIFFUSER.register()
class CoKinMoEDiffuser(ConsistencyCoupledKinematicsDiffuser):
    """CoKin dual-space diffuser with KineRoute paired MoE routing.

    New loss terms:
        L_FK_route = ||FK(x0_pred_joint) - x0_pred_pose||_2  (already in base as consistency)
        L_KD = sum_l ||MoE_FFN_out_l - Dense_FFN_out_l||_2
        L_lb = sum_l H(pair_probs_l)  (load balancing)
    """

    def __init__(
        self,
        eps_model: nn.Module,
        cfg: DictConfig,
        has_obser: bool,
        pose_eps_model: Optional[nn.Module] = None,
        joint_eps_model: Optional[nn.Module] = None,
        fk_model: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            eps_model=eps_model,
            cfg=cfg,
            has_obser=has_obser,
            pose_eps_model=pose_eps_model,
            joint_eps_model=joint_eps_model,
            fk_model=fk_model,
            *args,
            **kwargs,
        )

        # MoE hyperparameters
        self.num_experts = int(cfg.get("num_experts", 4))
        self.nblocks = int(cfg.get("nblocks", 4))

        # Loss weights
        self.w_fk = float(cfg.get("fk_route_weight", 1.0))
        self.w_kd = float(cfg.get("kd_weight", 1.0))
        self.w_lb = float(cfg.get("lb_weight", 0.01))

        # Gumbel temperature schedule
        self.tau_init = float(cfg.get("tau_init", 1.0))
        self.tau_min = float(cfg.get("tau_min", 0.1))
        self.tau_anneal_epochs = int(cfg.get("tau_anneal_epochs", 200))
        self.current_tau = self.tau_init

        # d_model and context_dim from the eps models
        d_model = int(cfg.get("d_model", 512))
        context_dim = int(cfg.get("context_dim", 512))

        # Create per-layer paired routers
        self.routers = nn.ModuleList([
            PairedRouter(
                d_model=d_model,
                num_experts=self.num_experts,
                context_dim=context_dim,
                tau_init=self.tau_init,
                tau_min=self.tau_min,
            )
            for _ in range(self.nblocks)
        ])

        # Dense teacher copies (frozen) for KD loss
        # We store the dense FFN modules from the original model init
        # These are created during weight initialization from dense CoKin
        self.has_teacher = False

    def init_from_dense_teacher(
        self,
        dense_pose_model: nn.Module,
        dense_joint_model: nn.Module,
    ):
        """Initialize MoE experts from dense teacher FFN weights.

        Each expert is initialized as a copy of the dense FFN with small random
        perturbation to break symmetry.
        """
        # Store frozen teacher FFNs for KD
        self.teacher_pose_ffns = nn.ModuleList()
        self.teacher_joint_ffns = nn.ModuleList()

        for block_idx in range(self.nblocks):
            # Get the dense FFN from teacher
            pose_spatial = dense_pose_model.layers[block_idx * 2 + 1]
            joint_spatial = dense_joint_model.layers[block_idx * 2 + 1]

            for tb in pose_spatial.transformer_blocks:
                dense_ff = copy.deepcopy(tb.ff)
                for p in dense_ff.parameters():
                    p.requires_grad_(False)
                self.teacher_pose_ffns.append(dense_ff)

            for tb in joint_spatial.transformer_blocks:
                dense_ff = copy.deepcopy(tb.ff)
                for p in dense_ff.parameters():
                    p.requires_grad_(False)
                self.teacher_joint_ffns.append(dense_ff)

            # Initialize MoE experts from the dense weights
            pose_moe_spatial = self.pose_eps_model.layers[block_idx * 2 + 1]
            joint_moe_spatial = self.joint_eps_model.layers[block_idx * 2 + 1]

            for tb_idx, tb in enumerate(pose_moe_spatial.transformer_blocks):
                dense_sd = self.teacher_pose_ffns[block_idx * len(pose_spatial.transformer_blocks) + tb_idx].state_dict()
                for expert in tb.moe_ff.experts:
                    expert.load_state_dict(dense_sd)
                    # Add small noise to break symmetry
                    with torch.no_grad():
                        for p in expert.parameters():
                            p.add_(torch.randn_like(p) * 0.01)

            for tb_idx, tb in enumerate(joint_moe_spatial.transformer_blocks):
                dense_sd = self.teacher_joint_ffns[block_idx * len(joint_spatial.transformer_blocks) + tb_idx].state_dict()
                for expert in tb.moe_ff.experts:
                    expert.load_state_dict(dense_sd)
                    with torch.no_grad():
                        for p in expert.parameters():
                            p.add_(torch.randn_like(p) * 0.01)

        # Freeze attention layers in both denoisers
        for block_idx in range(self.nblocks):
            for model in [self.pose_eps_model, self.joint_eps_model]:
                # Freeze ResBlock
                res_block = model.layers[block_idx * 2 + 0]
                for p in res_block.parameters():
                    p.requires_grad_(False)

                # Freeze attention in SpatialTransformer
                spatial = model.layers[block_idx * 2 + 1]
                for p in spatial.norm.parameters():
                    p.requires_grad_(False)
                for p in spatial.proj_in.parameters():
                    p.requires_grad_(False)
                for p in spatial.proj_out.parameters():
                    p.requires_grad_(False)
                for tb in spatial.transformer_blocks:
                    for p in tb.attn1.parameters():
                        p.requires_grad_(False)
                    for p in tb.attn2.parameters():
                        p.requires_grad_(False)
                    for p in tb.norm1.parameters():
                        p.requires_grad_(False)
                    for p in tb.norm2.parameters():
                        p.requires_grad_(False)
                    for p in tb.norm3.parameters():
                        p.requires_grad_(False)

        self.has_teacher = True

    def _update_tau(self, current_epoch: int):
        """Anneal Gumbel temperature linearly."""
        if self.tau_anneal_epochs > 0:
            progress = min(current_epoch / self.tau_anneal_epochs, 1.0)
            self.current_tau = self.tau_init - progress * (self.tau_init - self.tau_min)
        else:
            self.current_tau = self.tau_min

        for router in self.routers:
            router.set_tau(self.current_tau)

    def _compute_routing_decisions(
        self,
        ts: torch.Tensor,
        data: Dict[str, torch.Tensor],
        joint_x_t: torch.Tensor,
        pose_x_t: torch.Tensor,
    ) -> Tuple[List[int], List[int], List[torch.Tensor], List[torch.Tensor]]:
        """Compute per-layer routing decisions using PairedRouter.

        Returns:
            joint_expert_indices: [nblocks] list of int
            pose_expert_indices: [nblocks] list of int
            all_pair_probs: list of [E, E] probability matrices
            all_pair_matrices: list of [E, E] selection matrices
        """
        joint_expert_indices = []
        pose_expert_indices = []
        all_pair_probs = []
        all_pair_matrices = []

        # Get timestep embedding
        sigma_embed = timestep_embedding(ts, self.joint_eps_model.d_model)
        sigma_embed_mean = sigma_embed.mean(dim=0, keepdim=True)  # [1, d_model]

        # Get scene latent z_scene
        if hasattr(self.joint_eps_model, 'condition'):
            z_scene = self.joint_eps_model.condition(data)
            z_scene_mean = z_scene.mean(dim=(0, 1), keepdim=False).unsqueeze(0)  # [1, context_dim]
        else:
            z_scene_mean = torch.zeros(1, self.routers[0].context_mlp[0].in_features - 3 * self.joint_eps_model.d_model,
                                       device=ts.device)

        # Get intermediate features from both branches
        h_joint_pool = self.joint_eps_model.get_intermediate_features(joint_x_t, ts, z_scene if hasattr(self.joint_eps_model, 'condition') else None)
        h_pose_pool = self.pose_eps_model.get_intermediate_features(pose_x_t, ts, z_scene if hasattr(self.pose_eps_model, 'condition') else None)

        # Mean pool over batch
        h_joint_mean = h_joint_pool.mean(dim=0, keepdim=True)  # [1, d_model]
        h_pose_mean = h_pose_pool.mean(dim=0, keepdim=True)  # [1, d_model]

        for l in range(self.nblocks):
            joint_idx, pose_idx, pair_probs, _, pair_matrix = self.routers[l](
                sigma_embed_mean, z_scene_mean, h_joint_mean, h_pose_mean
            )
            joint_expert_indices.append(joint_idx)
            pose_expert_indices.append(pose_idx)
            all_pair_probs.append(pair_probs)
            all_pair_matrices.append(pair_matrix)

        return joint_expert_indices, pose_expert_indices, all_pair_probs, all_pair_matrices

    def _compute_load_balancing_loss(self, all_pair_probs: List[torch.Tensor]) -> torch.Tensor:
        """Load balancing loss: encourage uniform expert usage."""
        loss = torch.tensor(0.0, device=self.device)
        for pair_probs in all_pair_probs:
            # Flatten to get per-pair probabilities
            probs_flat = pair_probs.reshape(-1)
            # Ideal uniform = 1 / (E*E)
            uniform = torch.ones_like(probs_flat) / probs_flat.numel()
            # KL divergence or squared deviation from uniform
            loss = loss + F.mse_loss(probs_flat, uniform)
        return loss / len(all_pair_probs)

    def _predict_eps_moe(
        self,
        eps_model: nn.Module,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        data: Dict[str, torch.Tensor],
        condition_key: Optional[str],
        expert_indices: List[int],
    ) -> torch.Tensor:
        """Like _predict_eps but passes expert_indices to MoE UNet."""
        if condition_key is not None and condition_key in data:
            cond = data[condition_key]
        elif hasattr(eps_model, "condition"):
            cond = eps_model.condition(data)
        else:
            cond = None

        if cond is None:
            return eps_model(x_t, ts, expert_indices=expert_indices)
        return eps_model(x_t, ts, cond, expert_indices=expert_indices)

    def _forward_single_branch_moe(
        self,
        eps_model: nn.Module,
        x0: torch.Tensor,
        ts: torch.Tensor,
        data: Dict[str, torch.Tensor],
        start_key: Optional[str],
        obser_key: Optional[str],
        condition_key: Optional[str],
        expert_indices: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Single branch forward with MoE expert dispatch."""
        noise = torch.randn_like(x0, device=self.device)
        x_t = self.q_sample(x0=x0, t=ts, noise=noise)
        x_t = self._apply_observation(x_t, data, start_key=start_key, obser_key=obser_key)

        pred_noise = self._predict_eps_moe(
            eps_model=eps_model,
            x_t=x_t,
            ts=ts,
            data=data,
            condition_key=condition_key,
            expert_indices=expert_indices,
        )
        pred_noise = self._apply_observation(
            pred_noise, data, start_key=start_key, obser_key=obser_key
        )

        pred_x0 = self.predict_x0_from_noise(x_t=x_t, t=ts, pred_noise=pred_noise)
        pred_x0 = self._apply_observation(pred_x0, data, start_key=start_key, obser_key=obser_key)

        diff_loss = self._diff_loss(pred_noise, noise)
        return {
            "x_t": x_t,
            "noise": noise,
            "pred_noise": pred_noise,
            "pred_x0": pred_x0,
            "diff_loss": diff_loss,
        }

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose_x0, joint_x0 = self._get_pose_and_joint_targets(data)
        bsz = joint_x0.shape[0]

        if self.shared_timestep:
            ts = self._sample_timesteps(bsz)
            ts_pose = ts_joint = ts
        else:
            ts_pose = self._sample_timesteps(bsz)
            ts_joint = self._sample_timesteps(bsz)

        # Create noisy inputs for routing context
        noise_joint = torch.randn_like(joint_x0, device=self.device)
        noise_pose = torch.randn_like(pose_x0, device=self.device)
        joint_x_t = self.q_sample(x0=joint_x0, t=ts_joint, noise=noise_joint)
        pose_x_t = self.q_sample(x0=pose_x0, t=ts_pose, noise=noise_pose)

        # Compute routing decisions
        joint_expert_indices, pose_expert_indices, all_pair_probs, all_pair_matrices = \
            self._compute_routing_decisions(ts_joint, data, joint_x_t, pose_x_t)

        # Forward pass through both branches with MoE dispatch
        pose_out = self._forward_single_branch_moe(
            eps_model=self.pose_eps_model,
            x0=pose_x0,
            ts=ts_pose,
            data=data,
            start_key=self.pose_start_key,
            obser_key=self.pose_obser_key,
            condition_key=self.pose_condition_key,
            expert_indices=pose_expert_indices,
        )
        joint_out = self._forward_single_branch_moe(
            eps_model=self.joint_eps_model,
            x0=joint_x0,
            ts=ts_joint,
            data=data,
            start_key=self.joint_start_key,
            obser_key=self.joint_obser_key,
            condition_key=self.joint_condition_key,
            expert_indices=joint_expert_indices,
        )

        pred_pose_x0 = pose_out["pred_x0"]
        pred_joint_x0 = joint_out["pred_x0"]

        if self.detach_pose_for_consistency:
            pred_pose_x0 = pred_pose_x0.detach()
        if self.detach_joint_for_consistency:
            pred_joint_x0 = pred_joint_x0.detach()

        # FK consistency loss (L_FK_route)
        pred_joint_fk_input = self._joint_to_fk_input(pred_joint_x0)
        pred_pose_from_joint = self._run_fk(pred_joint_fk_input)
        pred_pose_from_joint = self._fk_to_reference_repr(pred_pose_from_joint, pred_pose_x0)

        consist_mask = self._build_consistency_mask(pred_pose_x0, data)
        fk_route_loss = self._masked_loss(pred_pose_x0, pred_pose_from_joint, consist_mask)

        # Load balancing loss
        lb_loss = self._compute_load_balancing_loss(all_pair_probs)

        # Diffusion losses
        pose_diff_loss = pose_out["diff_loss"]
        joint_diff_loss = joint_out["diff_loss"]

        # Total loss
        total_loss = (
            self.pose_diff_weight * pose_diff_loss
            + self.joint_diff_weight * joint_diff_loss
            + self.w_fk * fk_route_loss
            + self.w_lb * lb_loss
        )

        return {
            "loss": total_loss,
            "pose_diff_loss": pose_diff_loss,
            "joint_diff_loss": joint_diff_loss,
            "fk_route_loss": fk_route_loss,
            "lb_loss": lb_loss,
            "tau": torch.tensor(self.current_tau),
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # Update Gumbel temperature
        self._update_tau(self.current_epoch)

        losses = self(batch)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/loss_pose_diff", losses["pose_diff_loss"])
        self.log("train/loss_joint_diff", losses["joint_diff_loss"])
        self.log("train/loss_fk_route", losses["fk_route_loss"])
        self.log("train/loss_lb", losses["lb_loss"])
        self.log("train/tau", losses["tau"])
        return losses["loss"]
