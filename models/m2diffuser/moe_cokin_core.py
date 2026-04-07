from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def sinusoidal_timestep_embedding(
    timesteps: Tensor,
    dim: int,
    max_period: float = 10_000.0,
) -> Tensor:
    """Create sinusoidal embeddings from scalar timesteps."""
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps.to(dtype=torch.float32)

    half = dim // 2
    if half == 0:
        raise ValueError("dim must be >= 2 for sinusoidal embedding.")

    freq_exponent = -math.log(max_period) * torch.arange(
        half, device=timesteps.device, dtype=timesteps.dtype
    ) / max(half - 1, 1)
    freqs = torch.exp(freq_exponent)
    angles = timesteps[:, None] * freqs[None, :]

    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm: inject noise embedding into token features."""

    def __init__(self, d_model: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.modulation = nn.Linear(cond_dim, 2 * d_model)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        while shift.ndim < x.ndim:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)

        # AdaLN(x, c) = LN(x) * (1 + gamma(c)) + beta(c)
        return self.norm(x) * (1.0 + scale) + shift


@dataclass
class RouterOutput:
    logits: Tensor
    probs: Tensor
    topk_indices: Tensor
    topk_weights: Tensor


class SwishGLUExpertMLP(nn.Module):
    """Expert MLP with Swish-GLU activation."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.fc_in(x).chunk(2, dim=-1)
        # Swish-GLU(x) = value(x) * swish(gate(x))
        hidden = value * F.silu(gate)
        hidden = self.dropout(hidden)
        return self.fc_out(hidden)


class NoiseConditionedRouter(nn.Module):
    """Noise-only router: P = softmax(phi(sigma_t) @ W_R)."""

    def __init__(
        self,
        noise_emb_dim: int,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be >= 1.")
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = float(temperature)

        self.route_weight = nn.Parameter(torch.empty(noise_emb_dim, num_experts))
        nn.init.xavier_uniform_(self.route_weight)

    def forward(self, noise_clock_emb: Tensor) -> RouterOutput:
        if noise_clock_emb.ndim != 2:
            raise ValueError("noise_clock_emb must be [B, D_noise].")

        # Router does NOT consume token content X, only phi(sigma_t).
        logits = (noise_clock_emb @ self.route_weight) / self.temperature
        probs = torch.softmax(logits, dim=-1)

        topk_weights, topk_indices = torch.topk(probs, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return RouterOutput(
            logits=logits,
            probs=probs,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
        )


class NoiseConditionedMoE(nn.Module):
    """MoE FFN replacement controlled only by noise timestep embedding."""

    def __init__(
        self,
        d_model: int,
        expert_hidden_dim: int,
        noise_emb_dim: int,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.router = NoiseConditionedRouter(
            noise_emb_dim=noise_emb_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        self.experts = nn.ModuleList(
            [SwishGLUExpertMLP(d_model, expert_hidden_dim, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: Tensor, noise_clock_emb: Tensor) -> Tuple[Tensor, RouterOutput]:
        routing = self.router(noise_clock_emb)
        mixed = torch.zeros_like(x)

        # Sparse top-k expert dispatch with batch-wise shared routing weights.
        unique_experts = torch.unique(routing.topk_indices).tolist()
        for expert_idx in unique_experts:
            mask = (routing.topk_indices == expert_idx).to(dtype=x.dtype)
            coeff = (routing.topk_weights * mask).sum(dim=-1)  # [B]
            if float(coeff.abs().sum().item()) == 0.0:
                continue
            expert_out = self.experts[int(expert_idx)](x)
            mixed = mixed + coeff[:, None, None] * expert_out

        return mixed, routing


class MoE_CoKin_TransformerBlock(nn.Module):
    """Section 4.1: Scene CA + NCSA + independent noise-conditioned MoE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        noise_emb_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.noise_embed = nn.Sequential(
            nn.Linear(d_model, noise_emb_dim),
            nn.SiLU(),
            nn.Linear(noise_emb_dim, noise_emb_dim),
        )

        self.scene_norm = nn.LayerNorm(d_model)
        self.scene_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ncsa_adaln = AdaLayerNorm(d_model=d_model, cond_dim=noise_emb_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.pose_moe_adaln = AdaLayerNorm(d_model=d_model, cond_dim=noise_emb_dim)
        self.joint_moe_adaln = AdaLayerNorm(d_model=d_model, cond_dim=noise_emb_dim)

        # Independent scheduling for dual-space branches.
        self.pose_moe = NoiseConditionedMoE(
            d_model=d_model,
            expert_hidden_dim=expert_hidden_dim,
            noise_emb_dim=noise_emb_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )
        self.joint_moe = NoiseConditionedMoE(
            d_model=d_model,
            expert_hidden_dim=expert_hidden_dim,
            noise_emb_dim=noise_emb_dim,
            num_experts=num_experts,
            top_k=top_k,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def _build_noise_clock_embedding(self, t: Tensor, batch_size: int) -> Tensor:
        if t.ndim == 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1:
            raise ValueError("t must be scalar or [B].")

        t_emb = sinusoidal_timestep_embedding(timesteps=t, dim=self.d_model)
        noise_clock = self.noise_embed(t_emb)

        if noise_clock.shape[0] == 1 and batch_size > 1:
            noise_clock = noise_clock.expand(batch_size, -1)
        if noise_clock.shape[0] != batch_size:
            raise ValueError("Batch mismatch between X and noise timestep embedding.")
        return noise_clock

    def forward(
        self,
        x: Tensor,
        scene_tokens: Tensor,
        t: Tensor,
        branch: Literal["pose", "joint"],
    ) -> Tuple[Tensor, RouterOutput]:
        if x.ndim != 3:
            raise ValueError("x must be [B, L, D].")
        if scene_tokens.ndim != 3:
            raise ValueError("scene_tokens must be [B, N_scene, D].")

        noise_clock = self._build_noise_clock_embedding(t=t, batch_size=x.shape[0])

        # (1) Scene Cross-Attention: Q=X, K=V=S.
        x_scene_in = self.scene_norm(x)
        scene_ctx, _ = self.scene_cross_attn(
            query=x_scene_in,
            key=scene_tokens,
            value=scene_tokens,
            need_weights=False,
        )
        x = x + self.dropout(scene_ctx)

        # (2) NCSA: inject phi(sigma_t) via AdaLN before self-attention.
        x_ncsa_in = self.ncsa_adaln(x, noise_clock)
        self_ctx, _ = self.self_attn(
            query=x_ncsa_in,
            key=x_ncsa_in,
            value=x_ncsa_in,
            need_weights=False,
        )
        x = x + self.dropout(self_ctx)

        # (3) Independent noise-conditioned MoE FFN replacement.
        if branch == "pose":
            x_moe_in = self.pose_moe_adaln(x, noise_clock)
            moe_out, route_info = self.pose_moe(x_moe_in, noise_clock)
        elif branch == "joint":
            x_moe_in = self.joint_moe_adaln(x, noise_clock)
            moe_out, route_info = self.joint_moe(x_moe_in, noise_clock)
        else:
            raise ValueError(f"Unsupported branch: {branch}")

        x = x + self.dropout(moe_out)
        return x, route_info


class MoECoKin_LossCriterion(nn.Module):
    """Section 4.2.2: diffusion + load-balance + kinematic consistency."""

    def __init__(
        self,
        w_diff_pose: float = 1.0,
        w_diff_joint: float = 1.0,
        w_lb_pose: float = 1e-2,
        w_lb_joint: float = 1e-2,
        w_consistency: float = 1.0,
    ) -> None:
        super().__init__()
        self.w_diff_pose = float(w_diff_pose)
        self.w_diff_joint = float(w_diff_joint)
        self.w_lb_pose = float(w_lb_pose)
        self.w_lb_joint = float(w_lb_joint)
        self.w_consistency = float(w_consistency)

    @staticmethod
    def _load_balance_loss(route: RouterOutput) -> Tensor:
        if route.probs.ndim != 2:
            raise ValueError("route.probs must be [B, E].")

        num_experts = route.probs.shape[-1]
        dispatch = torch.zeros_like(route.probs)
        dispatch.scatter_add_(1, route.topk_indices, route.topk_weights)

        # Switch-style auxiliary loss: L_LB = E * sum(mean(p_e) * mean(load_e)).
        importance = route.probs.mean(dim=0)
        load = dispatch.mean(dim=0)
        return num_experts * torch.sum(importance * load)

    def forward(
        self,
        pred_eps_pose: Tensor,
        target_eps_pose: Tensor,
        pred_eps_joint: Tensor,
        target_eps_joint: Tensor,
        tau_pose_hat: Tensor,
        tau_joint_hat: Tensor,
        phi_kinematics: Callable[[Tensor], Tensor],
        route_pose: RouterOutput,
        route_joint: RouterOutput,
    ) -> Dict[str, Tensor]:
        # Diffusion denoising losses: L_diff_pose and L_diff_joint.
        l_diff_pose = F.mse_loss(pred_eps_pose, target_eps_pose)
        l_diff_joint = F.mse_loss(pred_eps_joint, target_eps_joint)

        # Router load-balancing losses for two independent branches.
        l_lb_pose = self._load_balance_loss(route_pose)
        l_lb_joint = self._load_balance_loss(route_joint)

        tau_pose_from_joint = phi_kinematics(tau_joint_hat)
        # Kinematic consistency: ||tau_pose_hat - Phi(tau_joint_hat)||^2_2.
        l_consist = F.mse_loss(tau_pose_hat, tau_pose_from_joint)

        l_total = (
            self.w_diff_pose * l_diff_pose
            + self.w_diff_joint * l_diff_joint
            + self.w_lb_pose * l_lb_pose
            + self.w_lb_joint * l_lb_joint
            + self.w_consistency * l_consist
        )
        return {
            "L_total": l_total,
            "L_diff_pose": l_diff_pose,
            "L_diff_joint": l_diff_joint,
            "L_LB_pose": l_lb_pose,
            "L_LB_joint": l_lb_joint,
            "L_consist": l_consist,
        }


class CAMOGG_Sampler(nn.Module):
    """Section 4.3: cached expert fusion + C-AMOGG projected gradient."""

    def __init__(
        self,
        top_k: int = 2,
        kappa: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.kappa = float(kappa)
        self.eps = float(eps)

        self.cached_experts = nn.ModuleDict()
        self.cached_route: Dict[int, Tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _clone_like_expert(reference: SwishGLUExpertMLP) -> SwishGLUExpertMLP:
        cloned = SwishGLUExpertMLP(
            d_model=reference.d_model,
            hidden_dim=reference.hidden_dim,
            dropout=0.0,
        )
        return cloned.to(
            device=reference.fc_in.weight.device,
            dtype=reference.fc_in.weight.dtype,
        )

    def _fuse_experts(
        self,
        experts: Sequence[SwishGLUExpertMLP],
        expert_indices: Tensor,
        expert_weights: Tensor,
    ) -> SwishGLUExpertMLP:
        if expert_indices.ndim != 1 or expert_weights.ndim != 1:
            raise ValueError("expert_indices and expert_weights must be 1-D.")
        if expert_indices.shape[0] != expert_weights.shape[0]:
            raise ValueError("expert_indices and expert_weights length mismatch.")

        fused = self._clone_like_expert(experts[0])
        with torch.no_grad():
            fused.fc_in.weight.zero_()
            fused.fc_in.bias.zero_()
            fused.fc_out.weight.zero_()
            fused.fc_out.bias.zero_()

            for idx, weight in zip(expert_indices.tolist(), expert_weights):
                expert = experts[int(idx)]
                w = weight.to(dtype=fused.fc_in.weight.dtype, device=fused.fc_in.weight.device)

                # Expert caching fusion: W_fused = sum_i r_i * W_i.
                fused.fc_in.weight.add_(w * expert.fc_in.weight)
                fused.fc_in.bias.add_(w * expert.fc_in.bias)
                fused.fc_out.weight.add_(w * expert.fc_out.weight)
                fused.fc_out.bias.add_(w * expert.fc_out.bias)
        return fused

    def build_expert_cache(
        self,
        precomputed_route_probs: Tensor,
        experts: Sequence[SwishGLUExpertMLP],
    ) -> None:
        """Precompute static fused experts for each denoising timestep."""
        if precomputed_route_probs.ndim == 3:
            route_probs = precomputed_route_probs.mean(dim=0)
        elif precomputed_route_probs.ndim == 2:
            route_probs = precomputed_route_probs
        else:
            raise ValueError("precomputed_route_probs must be [T, E] or [B, T, E].")

        self.cached_experts = nn.ModuleDict()
        self.cached_route = {}

        for t_idx in range(route_probs.shape[0]):
            probs_t = route_probs[t_idx]
            topk_weights, topk_indices = torch.topk(probs_t, k=self.top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum().clamp_min(1e-8)

            fused_expert = self._fuse_experts(
                experts=experts,
                expert_indices=topk_indices,
                expert_weights=topk_weights,
            )
            self.cached_experts[str(t_idx)] = fused_expert.eval()
            self.cached_route[t_idx] = (
                topk_indices.detach().cpu(),
                topk_weights.detach().cpu(),
            )

    def cached_expert_forward(self, x: Tensor, t: int) -> Tensor:
        key = str(int(t))
        if key not in self.cached_experts:
            raise KeyError(f"No cached expert found for timestep {t}.")
        return self.cached_experts[key](x)

    @staticmethod
    def _reduce_objective(obj: Tensor) -> Tensor:
        if obj.ndim == 0:
            return obj
        if obj.ndim == 1:
            return obj.sum()
        raise ValueError("Objective must be scalar or [B].")

    def _objective_gradient(
        self,
        x: Tensor,
        objective_fn: Callable[[Tensor, Optional[Dict[str, Tensor]]], Tensor],
        context: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = objective_fn(x_in, context)
            if not torch.is_tensor(obj):
                obj = torch.as_tensor(obj, dtype=x_in.dtype, device=x_in.device)
            obj = self._reduce_objective(obj)
            grad = torch.autograd.grad(obj, x_in, create_graph=False, retain_graph=False)[0]
        return grad

    def _lambda_t(self, t: Tensor, total_steps: int, ref: Tensor) -> Tensor:
        denom = float(max(total_steps - 1, 1))
        u = t.to(dtype=ref.dtype, device=ref.device) / denom
        lam = torch.sigmoid(self.kappa * (u - 0.5))
        while lam.ndim < ref.ndim:
            lam = lam.unsqueeze(-1)
        return lam

    def _project_conflict_averse(self, g1: Tensor, g2: Tensor) -> Tensor:
        reduce_dims = tuple(range(1, g1.ndim))
        dot = torch.sum(g1 * g2, dim=reduce_dims, keepdim=True)
        norm_g1_sq = torch.sum(g1 * g1, dim=reduce_dims, keepdim=True).clamp_min(self.eps)

        # If g1^T g2 < 0: g2* = g2 - ((g2^T g1)/||g1||^2) g1
        g2_projected = g2 - (dot / norm_g1_sq) * g1
        return torch.where(dot < 0, g2_projected, g2)

    def compute_camogg_gradient(
        self,
        x: Tensor,
        t: Tensor,
        total_steps: int,
        chamfer_objective: Callable[[Tensor, Optional[Dict[str, Tensor]]], Tensor],
        sdf_objective: Callable[[Tensor, Optional[Dict[str, Tensor]]], Tensor],
        context: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        # g1: gradient of task objective (e.g., Chamfer distance).
        g1 = self._objective_gradient(x, chamfer_objective, context)
        # g2: gradient of collision penalty objective (e.g., SDF).
        g2 = self._objective_gradient(x, sdf_objective, context)

        g2_star = self._project_conflict_averse(g1, g2)
        lam = self._lambda_t(t=t, total_steps=total_steps, ref=x)

        # Final composed gradient: g = lambda(t) * g1 + (1 - lambda(t)) * g2*.
        return lam * g1 + (1.0 - lam) * g2_star

    def guided_step(
        self,
        x: Tensor,
        t: Tensor,
        total_steps: int,
        step_size: float,
        chamfer_objective: Callable[[Tensor, Optional[Dict[str, Tensor]]], Tensor],
        sdf_objective: Callable[[Tensor, Optional[Dict[str, Tensor]]], Tensor],
        context: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        grad = self.compute_camogg_gradient(
            x=x,
            t=t,
            total_steps=total_steps,
            chamfer_objective=chamfer_objective,
            sdf_objective=sdf_objective,
            context=context,
        )
        return x - float(step_size) * grad
