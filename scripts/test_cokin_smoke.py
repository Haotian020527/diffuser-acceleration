import argparse
import os
import sys
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Ensure project root is importable when running as:
# `python scripts/test_cokin_smoke.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.m2diffuser.cokin import ConsistencyCoupledKinematicsDiffuser


class DummyEpsModel(nn.Module):
    """Minimal epsilon model matching M2Diffuser-like interface."""

    def __init__(self, d_x: int, cond_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.d_x = d_x
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(d_x + 1 + cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_x),
        )

    def condition(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        return data["dummy_cond"]

    def forward(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        cond: torch.Tensor = None,
    ) -> torch.Tensor:
        squeeze_time = False
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)
            squeeze_time = True

        bsz, horizon, _ = x_t.shape
        t_embed = ts.float().view(bsz, 1, 1) / 1000.0

        if cond is None:
            cond_feat = torch.zeros(
                bsz, 1, self.cond_dim, dtype=x_t.dtype, device=x_t.device
            )
        else:
            if cond.ndim == 2:
                cond = cond.unsqueeze(1)
            cond_feat = cond.mean(dim=1, keepdim=True)
            if cond_feat.shape[-1] > self.cond_dim:
                cond_feat = cond_feat[..., : self.cond_dim]
            elif cond_feat.shape[-1] < self.cond_dim:
                pad = self.cond_dim - cond_feat.shape[-1]
                cond_feat = torch.cat(
                    [
                        cond_feat,
                        torch.zeros(
                            bsz,
                            1,
                            pad,
                            dtype=cond_feat.dtype,
                            device=cond_feat.device,
                        ),
                    ],
                    dim=-1,
                )

        cond_feat = cond_feat.expand(bsz, horizon, self.cond_dim)
        model_in = torch.cat([x_t, t_embed.expand(bsz, horizon, 1), cond_feat], dim=-1)
        out = self.net(model_in)

        if squeeze_time:
            out = out.squeeze(1)
        return out


class SimpleFK(nn.Module):
    """Differentiable FK stub: joints -> [xyz, quaternion]."""

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        if joints.ndim != 2:
            raise ValueError(f"SimpleFK expects [N, dof], got {tuple(joints.shape)}")

        xyz = joints[:, :3]
        quat_raw = torch.stack(
            [
                joints[:, 3],
                joints[:, 4],
                joints[:, 5],
                torch.ones_like(joints[:, 3]),
            ],
            dim=-1,
        )
        quat = quat_raw / torch.linalg.norm(quat_raw, dim=-1, keepdim=True).clamp(
            min=1e-8
        )
        return torch.cat([xyz, quat], dim=-1)


def build_cfg():
    return OmegaConf.create(
        {
            "timesteps": 8,
            "schedule_cfg": {
                "beta": [1e-4, 2e-2],
                "beta_schedule": "linear",
                "s": 0.008,
            },
            "rand_t_type": "all",
            "lr": 1e-3,
            "loss_type": "l2",
            "shared_timestep": True,
            "pose_diff_weight": 1.0,
            "joint_diff_weight": 1.0,
            "consistency_weight": 1.0,
            "auto_pose_target_from_joint": True,
            "joint_unnormalize_for_fk": False,
            "pose": {
                "x_key": "pose_x",
            },
            "joint": {
                "x_key": "x",
            },
        }
    )


def assert_grads(module: nn.Module, module_name: str) -> None:
    grads = [p.grad for p in module.parameters() if p.requires_grad]
    non_none = [g for g in grads if g is not None]
    if len(non_none) == 0:
        raise RuntimeError(f"{module_name} has no gradients.")
    if not all(torch.isfinite(g).all().item() for g in non_none):
        raise RuntimeError(f"{module_name} gradients contain NaN/Inf.")


def run_case(
    model: ConsistencyCoupledKinematicsDiffuser,
    device: torch.device,
    batch_size: int,
    horizon: int,
    joint_dim: int,
    include_pose_target: bool,
) -> None:
    model.train()
    optimizer = model.configure_optimizers()
    optimizer.zero_grad()

    batch = {
        "x": torch.randn(batch_size, horizon, joint_dim, device=device),
        "dummy_cond": torch.randn(batch_size, 4, 16, device=device),
    }

    if include_pose_target:
        with torch.no_grad():
            bsz, t, d = batch["x"].shape
            pose = model.fk_model(batch["x"].reshape(-1, d)).reshape(bsz, t, 7)
        batch["pose_x"] = pose

    losses = model(batch)
    required_keys = {"loss", "pose_diff_loss", "joint_diff_loss", "consistency_loss"}
    if set(losses.keys()) != required_keys:
        raise RuntimeError(f"Unexpected loss keys: {list(losses.keys())}")

    for name, val in losses.items():
        if val.ndim != 0:
            raise RuntimeError(f"{name} is not scalar: shape={tuple(val.shape)}")
        if not torch.isfinite(val).item():
            raise RuntimeError(f"{name} is NaN/Inf.")

    losses["loss"].backward()
    assert_grads(model.pose_eps_model, "pose_eps_model")
    assert_grads(model.joint_eps_model, "joint_eps_model")
    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for CoKin dual-space diffuser.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--joint-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    cfg = build_cfg()
    pose_eps = DummyEpsModel(d_x=7).to(device)
    joint_eps = DummyEpsModel(d_x=args.joint_dim).to(device)
    fk_model = SimpleFK().to(device)

    model = ConsistencyCoupledKinematicsDiffuser(
        eps_model=pose_eps,
        cfg=cfg,
        has_obser=False,
        pose_eps_model=pose_eps,
        joint_eps_model=joint_eps,
        fk_model=fk_model,
    ).to(device)

    run_case(
        model=model,
        device=device,
        batch_size=args.batch_size,
        horizon=args.horizon,
        joint_dim=args.joint_dim,
        include_pose_target=False,  # test auto pose target from FK
    )
    run_case(
        model=model,
        device=device,
        batch_size=args.batch_size,
        horizon=args.horizon,
        joint_dim=args.joint_dim,
        include_pose_target=True,  # test explicit pose supervision path
    )

    print("[PASS] CoKin smoke test passed.")


if __name__ == "__main__":
    main()
