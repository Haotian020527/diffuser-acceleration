import copy
from typing import Any, Callable, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from models.base import DIFFUSER
from models.m2diffuser.schedule import make_schedule_ddpm
from utils.torch_urdf import TorchURDF


def _reshape_timestep_buffer(buffer: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gather timestep coefficients and reshape to broadcast with x."""
    batch = t.shape[0]
    return buffer[t].reshape(batch, *((1,) * (x.ndim - 1)))


class TorchURDFFKAdapter(nn.Module):
    """Differentiable FK adapter based on TorchURDF.

    Input:
        joints: [N, dof]
    Output:
        end-effector poses as homogeneous matrices [N, 4, 4]
    """

    def __init__(self, urdf_path: str, eef_frame: str = "end_effector_link") -> None:
        super().__init__()
        self.urdf_path = urdf_path
        self.eef_frame = eef_frame
        self._robots_by_device: Dict[str, Any] = {}

    def _get_robot(self, device: torch.device):
        key = str(device)
        if key not in self._robots_by_device:
            self._robots_by_device[key] = TorchURDF.load(
                self.urdf_path,
                lazy_load_meshes=True,
                device=device,
            )
        return self._robots_by_device[key]

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        if joints.ndim != 2:
            raise ValueError(f"TorchURDFFKAdapter expects [N, dof], got {tuple(joints.shape)}")
        robot = self._get_robot(joints.device)
        fk = robot.link_fk_batch(joints, use_names=True)
        return fk[self.eef_frame]


@DIFFUSER.register()
class ConsistencyCoupledKinematicsDiffuser(pl.LightningModule):
    """Dual-space DDPM trainer for CoKin.

    Two denoisers are trained jointly:
    - pose diffuser   : eps_pose(x_t_pose, t, c_pose)
    - joint diffuser  : eps_joint(x_t_joint, t, c_joint)

    with consistency coupling:
        L_consist = ||x0_pose_hat - Phi(x0_joint_hat)||^2

    Notes:
    - The module focuses on training forward pass and loss coupling.
    - `fk_model` can be any differentiable callable / nn.Module that maps
      joint trajectories to end-effector poses.
    - If pose supervision is absent, pose target can be generated from
      joint target through FK (enabled by default).
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
        super().__init__()

        # Keep signature compatible with existing create_diffuser API.
        pose_net = pose_eps_model if pose_eps_model is not None else eps_model
        if joint_eps_model is not None:
            joint_net = joint_eps_model
        else:
            joint_net = copy.deepcopy(pose_net)

        self.pose_eps_model = pose_net
        self.joint_eps_model = joint_net

        # FK model is optional at init. If missing, we can build from cfg.fk.
        self.fk_model = fk_model

        self.timesteps = int(cfg.timesteps)
        self.rand_t_type = cfg.rand_t_type
        self.lr = float(cfg.lr)
        self.has_observation = has_obser

        pose_cfg = cfg.pose if "pose" in cfg else {}
        joint_cfg = cfg.joint if "joint" in cfg else {}

        self.pose_x_key = pose_cfg.get("x_key", "pose_x")
        self.joint_x_key = joint_cfg.get("x_key", "x")

        # For joint branch, default to existing M2Diffuser observation keys.
        # Pose branch defaults to None so it will not accidentally reuse joint starts.
        self.pose_start_key = pose_cfg.get("start_key", None)
        self.pose_obser_key = pose_cfg.get("obser_key", None)
        self.joint_start_key = joint_cfg.get("start_key", "start")
        self.joint_obser_key = joint_cfg.get("obser_key", "obser")

        self.pose_condition_key = pose_cfg.get("condition_key", None)
        self.joint_condition_key = joint_cfg.get("condition_key", None)

        self.shared_timestep = bool(cfg.get("shared_timestep", True))
        self.pose_diff_weight = float(cfg.get("pose_diff_weight", 1.0))
        self.joint_diff_weight = float(cfg.get("joint_diff_weight", 1.0))
        self.consistency_weight = float(cfg.get("consistency_weight", 1.0))
        self.pose_lb_weight = float(cfg.get("pose_lb_weight", 0.0))
        self.joint_lb_weight = float(cfg.get("joint_lb_weight", 0.0))

        self.detach_pose_for_consistency = bool(cfg.get("detach_pose_for_consistency", False))
        self.detach_joint_for_consistency = bool(cfg.get("detach_joint_for_consistency", False))
        self.ignore_observation_in_consistency = bool(cfg.get("ignore_observation_in_consistency", True))

        # If pose GT is absent, derive it from joint GT via FK.
        self.auto_pose_target_from_joint = bool(cfg.get("auto_pose_target_from_joint", True))

        # Optional normalized-joint -> real-joint mapping before FK.
        self.joint_unnormalize_for_fk = bool(cfg.get("joint_unnormalize_for_fk", False))
        norm_range = cfg.get("joint_norm_range", [-1.0, 1.0])
        self.joint_norm_low = float(norm_range[0])
        self.joint_norm_high = float(norm_range[1])
        joint_limits = cfg.get("joint_limits", None)
        if joint_limits is not None:
            self.register_buffer("joint_limits", torch.as_tensor(joint_limits, dtype=torch.float32))
        else:
            self.joint_limits = None  # type: ignore[assignment]

        # If pose target is generated from FK, select output format from pose net dim.
        inferred_pose_dim = getattr(self.pose_eps_model, "d_x", None)
        self.pose_target_dim = int(cfg.get("pose_target_dim", inferred_pose_dim if inferred_pose_dim is not None else 7))
        self.pose_quat_format = str(cfg.get("pose_quat_format", "xyzw"))

        for k, v in make_schedule_ddpm(self.timesteps, **cfg.schedule_cfg).items():
            self.register_buffer(k, v)

        loss_type = cfg.loss_type
        if loss_type not in ("l1", "l2"):
            raise ValueError(f"Unsupported loss type: {loss_type}")
        self.loss_type = loss_type

        # Try to build FK from config when no external FK model is passed.
        if self.fk_model is None:
            self.fk_model = self._build_fk_from_cfg(cfg)

    @property
    def device(self) -> torch.device:
        return self.betas.device

    def set_fk_model(self, fk_model: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.fk_model = fk_model

    def _build_fk_from_cfg(self, cfg: DictConfig) -> Optional[nn.Module]:
        fk_cfg = cfg.get("fk", None)
        if fk_cfg is None:
            return None
        if not bool(fk_cfg.get("enabled", True)):
            return None

        fk_type = str(fk_cfg.get("type", "torch_urdf"))
        if fk_type != "torch_urdf":
            raise ValueError(f"Unsupported fk.type: {fk_type}")

        urdf_path = fk_cfg.get("urdf_path", None)
        if urdf_path is None:
            # Fall back to MecKinova URDF path in the current codebase.
            from env.agent.mec_kinova import MecKinova

            urdf_path = str(MecKinova.urdf_path)

        eef_frame = str(fk_cfg.get("eef_frame", "end_effector_link"))
        return TorchURDFFKAdapter(urdf_path=urdf_path, eef_frame=eef_frame)

    def _sample_timesteps(self, batch_size: int) -> torch.Tensor:
        if self.rand_t_type == "all":
            ts = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        elif self.rand_t_type == "half":
            ts = torch.randint(0, self.timesteps, ((batch_size + 1) // 2,), device=self.device)
            if batch_size % 2 == 1:
                ts = torch.cat([ts, self.timesteps - ts[:-1] - 1], dim=0).long()
            else:
                ts = torch.cat([ts, self.timesteps - ts - 1], dim=0).long()
        else:
            raise ValueError(f"Unsupported rand_t_type: {self.rand_t_type}")
        return ts

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _reshape_timestep_buffer(self.sqrt_alphas_cumprod, t, x0) * x0
            + _reshape_timestep_buffer(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise
        )

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        return (
            _reshape_timestep_buffer(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t
            - _reshape_timestep_buffer(self.sqrt_recipm1_alphas_cumprod, t, x_t) * pred_noise
        )

    def _apply_observation(
        self,
        x: torch.Tensor,
        data: Dict[str, torch.Tensor],
        start_key: Optional[str],
        obser_key: Optional[str],
    ) -> torch.Tensor:
        if not self.has_observation:
            return x
        if start_key is None or start_key not in data:
            return x

        start = data[start_key]
        if start.ndim != x.ndim or start.shape[-1] != x.shape[-1]:
            return x

        length = start.shape[1]
        x[:, :length, :] = start[:, :length, :].clone()

        if obser_key is not None and obser_key in data:
            obser = data[obser_key]
            if obser.ndim == x.ndim and obser.shape[-1] == x.shape[-1]:
                o_len = obser.shape[1]
                x[:, length : length + o_len, :] = obser.clone()
        return x

    def _predict_eps(
        self,
        eps_model: nn.Module,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        data: Dict[str, torch.Tensor],
        condition_key: Optional[str],
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        if condition_key is not None and condition_key in data:
            cond = data[condition_key]
        elif hasattr(eps_model, "condition"):
            cond = eps_model.condition(data)
        else:
            cond = None

        output: Any
        if cond is None:
            output = eps_model(x_t, ts)
        else:
            output = eps_model(x_t, ts, cond)

        route = getattr(eps_model, "latest_router_output", None)
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError("Tuple model output must be (pred_noise, route).")
            pred_noise, route = output
            if not torch.is_tensor(pred_noise):
                raise ValueError("First tuple item from eps_model must be torch.Tensor.")
            return pred_noise, route

        if not torch.is_tensor(output):
            raise ValueError("eps_model output must be Tensor or (Tensor, route).")
        return output, route

    def _load_balance_loss(self, route: Optional[Any]) -> torch.Tensor:
        if route is None:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        if not hasattr(route, "probs"):
            return torch.zeros((), dtype=torch.float32, device=self.device)
        if not hasattr(route, "topk_indices") or not hasattr(route, "topk_weights"):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        probs = route.probs
        topk_indices = route.topk_indices
        topk_weights = route.topk_weights

        if probs.ndim != 2:
            return torch.zeros((), dtype=torch.float32, device=probs.device)

        dispatch = torch.zeros_like(probs)
        dispatch.scatter_add_(1, topk_indices, topk_weights)

        num_experts = probs.shape[-1]
        importance = probs.mean(dim=0)
        load = dispatch.mean(dim=0)
        return num_experts * torch.sum(importance * load)

    def _diff_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target)
        return F.mse_loss(pred, target)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if self.loss_type == "l1":
            err = torch.abs(pred - target)
        else:
            err = (pred - target) ** 2

        if mask is None:
            return err.mean()
        err = err * mask
        denom = mask.sum().clamp(min=1.0)
        return err.sum() / denom

    def _matrix_to_quaternion_wxyz(self, matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix (...,3,3) to quaternion (...,4) in wxyz order."""
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Invalid rotation matrix shape: {matrix.shape}")

        batch_shape = matrix.shape[:-2]
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
            matrix.reshape(batch_shape + (9,)), dim=-1
        )

        q_abs = torch.sqrt(
            torch.clamp(
                torch.stack(
                    [
                        1.0 + m00 + m11 + m22,
                        1.0 + m00 - m11 - m22,
                        1.0 - m00 + m11 - m22,
                        1.0 - m00 - m11 + m22,
                    ],
                    dim=-1,
                ),
                min=0.0,
            )
        )

        quat_by_rijk = torch.stack(
            [
                torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
                torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
                torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
                torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
            ],
            dim=-2,
        )

        floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
        quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
        idx = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
        return quat_candidates[idx].reshape(batch_shape + (4,))

    def _normalize_quaternion_sign(self, quat: torch.Tensor) -> torch.Tensor:
        if self.pose_quat_format == "xyzw":
            scalar = quat[..., -1:]
        elif self.pose_quat_format == "wxyz":
            scalar = quat[..., :1]
        else:
            raise ValueError(f"Unsupported pose_quat_format: {self.pose_quat_format}")

        mask = scalar >= 0
        return torch.where(mask, quat, -quat)

    def _fk_to_pose7(self, fk_pose: torch.Tensor) -> torch.Tensor:
        if fk_pose.ndim >= 2 and fk_pose.shape[-1] == 7:
            quat = self._normalize_quaternion_sign(fk_pose[..., 3:])
            return torch.cat([fk_pose[..., :3], quat], dim=-1)

        matrix = self._fk_to_matrix(fk_pose)
        rot = matrix[..., :3, :3]
        xyz = matrix[..., :3, 3]
        quat_wxyz = self._matrix_to_quaternion_wxyz(rot)
        if self.pose_quat_format == "xyzw":
            quat = torch.cat([quat_wxyz[..., 1:], quat_wxyz[..., :1]], dim=-1)
        else:
            quat = quat_wxyz
        quat = self._normalize_quaternion_sign(quat)
        return torch.cat([xyz, quat], dim=-1)

    def _quaternion_to_matrix_wxyz(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion (...,4) in wxyz order to rotation matrix (...,3,3)."""
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp(min=1e-8)
        w, x, y, z = quat.unbind(dim=-1)

        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z
        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        m00 = ww + xx - yy - zz
        m01 = 2 * (xy - wz)
        m02 = 2 * (xz + wy)
        m10 = 2 * (xy + wz)
        m11 = ww - xx + yy - zz
        m12 = 2 * (yz - wx)
        m20 = 2 * (xz - wy)
        m21 = 2 * (yz + wx)
        m22 = ww - xx - yy + zz

        return torch.stack(
            [
                torch.stack([m00, m01, m02], dim=-1),
                torch.stack([m10, m11, m12], dim=-1),
                torch.stack([m20, m21, m22], dim=-1),
            ],
            dim=-2,
        )

    def _fk_to_matrix(self, fk_pose: torch.Tensor) -> torch.Tensor:
        if fk_pose.ndim >= 2 and fk_pose.shape[-2:] == (4, 4):
            return fk_pose
        if fk_pose.ndim >= 2 and fk_pose.shape[-1] == 16:
            return fk_pose.reshape(*fk_pose.shape[:-1], 4, 4)
        if fk_pose.ndim >= 2 and fk_pose.shape[-1] == 7:
            xyz = fk_pose[..., :3]
            quat = fk_pose[..., 3:]
            if self.pose_quat_format == "xyzw":
                quat_wxyz = torch.cat([quat[..., -1:], quat[..., :3]], dim=-1)
            elif self.pose_quat_format == "wxyz":
                quat_wxyz = quat
            else:
                raise ValueError(f"Unsupported pose_quat_format: {self.pose_quat_format}")

            rot = self._quaternion_to_matrix_wxyz(quat_wxyz)
            matrix = torch.zeros(
                (*fk_pose.shape[:-1], 4, 4),
                dtype=fk_pose.dtype,
                device=fk_pose.device,
            )
            matrix[..., :3, :3] = rot
            matrix[..., :3, 3] = xyz
            matrix[..., 3, 3] = 1.0
            return matrix
        raise ValueError(
            "FK output must be one of: (...,7), (...,16), (...,4,4); "
            f"got shape={tuple(fk_pose.shape)}"
        )

    def _fk_to_reference_repr(self, fk_pose: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim >= 2 and fk_pose.shape == reference.shape:
            return fk_pose
        if reference.ndim >= 2 and reference.shape[-1] == 7:
            return self._fk_to_pose7(fk_pose)
        if reference.ndim >= 2 and reference.shape[-1] == 16:
            return self._fk_to_matrix(fk_pose).reshape(*reference.shape[:-1], 16)
        if reference.ndim >= 2 and reference.shape[-2:] == (4, 4):
            return self._fk_to_matrix(fk_pose)
        raise ValueError(
            f"Cannot align FK output shape {tuple(fk_pose.shape)} to pose reference shape {tuple(reference.shape)}"
        )

    def _joint_to_fk_input(self, joint_traj: torch.Tensor) -> torch.Tensor:
        if not self.joint_unnormalize_for_fk:
            return joint_traj
        if self.joint_limits is None:
            raise ValueError(
                "joint_unnormalize_for_fk=True but joint_limits is not set in diffuser cfg."
            )

        low = self.joint_limits[:, 0].to(joint_traj.dtype)
        high = self.joint_limits[:, 1].to(joint_traj.dtype)
        for _ in range(joint_traj.ndim - 1):
            low = low.unsqueeze(0)
            high = high.unsqueeze(0)
        return (joint_traj - self.joint_norm_low) * (high - low) / (
            self.joint_norm_high - self.joint_norm_low
        ) + low

    def _run_fk(self, joint_input: torch.Tensor) -> torch.Tensor:
        if self.fk_model is None:
            raise RuntimeError("fk_model is not set. Call set_fk_model(...) before training.")

        if joint_input.ndim == 3:
            bsz, horizon, dof = joint_input.shape
            flat_joint = joint_input.reshape(-1, dof)
            fk_out = self.fk_model(flat_joint)
            if fk_out.ndim == 2:
                return fk_out.reshape(bsz, horizon, -1)
            if fk_out.ndim == 3 and fk_out.shape[-2:] == (4, 4):
                return fk_out.reshape(bsz, horizon, 4, 4)
            if fk_out.ndim == 4 and fk_out.shape[0] == bsz and fk_out.shape[1] == horizon:
                return fk_out
            raise ValueError(
                "Unsupported FK output shape from flattened input: "
                f"{tuple(fk_out.shape)}"
            )

        return self.fk_model(joint_input)

    def _build_pose_target_from_joint(self, joint_x0: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            joint_fk_input = self._joint_to_fk_input(joint_x0)
            fk_pose = self._run_fk(joint_fk_input)
            if self.pose_target_dim == 7:
                return self._fk_to_pose7(fk_pose)
            if self.pose_target_dim == 16:
                return self._fk_to_matrix(fk_pose).reshape(*joint_x0.shape[:-1], 16)
            if fk_pose.ndim >= 2 and fk_pose.shape[-1] == self.pose_target_dim:
                return fk_pose
        raise ValueError(
            f"Cannot build pose target with pose_target_dim={self.pose_target_dim} "
            f"from FK output shape={tuple(fk_pose.shape)}"
        )

    def _get_pose_and_joint_targets(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.joint_x_key not in data:
            raise KeyError(f"Missing joint target key '{self.joint_x_key}' in batch.")
        joint_x0 = data[self.joint_x_key]

        if self.pose_x_key in data:
            pose_x0 = data[self.pose_x_key]
            return pose_x0, joint_x0

        if not self.auto_pose_target_from_joint:
            raise KeyError(
                f"Missing pose target key '{self.pose_x_key}' and auto_pose_target_from_joint=False."
            )
        pose_x0 = self._build_pose_target_from_joint(joint_x0)
        return pose_x0, joint_x0

    def _forward_single_branch(
        self,
        eps_model: nn.Module,
        x0: torch.Tensor,
        ts: torch.Tensor,
        data: Dict[str, torch.Tensor],
        start_key: Optional[str],
        obser_key: Optional[str],
        condition_key: Optional[str],
    ) -> Dict[str, Any]:
        noise = torch.randn_like(x0, device=self.device)
        x_t = self.q_sample(x0=x0, t=ts, noise=noise)
        x_t = self._apply_observation(x_t, data, start_key=start_key, obser_key=obser_key)

        pred_noise, route = self._predict_eps(
            eps_model=eps_model,
            x_t=x_t,
            ts=ts,
            data=data,
            condition_key=condition_key,
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
            "route": route,
        }

    def _build_consistency_mask(self, pose_ref: torch.Tensor, data: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if not self.ignore_observation_in_consistency:
            return None
        if not self.has_observation:
            return None
        if self.pose_start_key is None or self.pose_start_key not in data:
            return None

        mask = torch.ones_like(pose_ref)
        start = data[self.pose_start_key]
        if start.ndim == pose_ref.ndim and start.shape[-1] == pose_ref.shape[-1]:
            t = start.shape[1]
            mask[:, :t, :] = 0.0
            if self.pose_obser_key is not None and self.pose_obser_key in data:
                obser = data[self.pose_obser_key]
                if obser.ndim == pose_ref.ndim and obser.shape[-1] == pose_ref.shape[-1]:
                    o = obser.shape[1]
                    mask[:, t : t + o, :] = 0.0
        return mask

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pose_x0, joint_x0 = self._get_pose_and_joint_targets(data)
        bsz = joint_x0.shape[0]

        if self.shared_timestep:
            ts = self._sample_timesteps(bsz)
            ts_pose = ts_joint = ts
        else:
            ts_pose = self._sample_timesteps(bsz)
            ts_joint = self._sample_timesteps(bsz)

        pose_out = self._forward_single_branch(
            eps_model=self.pose_eps_model,
            x0=pose_x0,
            ts=ts_pose,
            data=data,
            start_key=self.pose_start_key,
            obser_key=self.pose_obser_key,
            condition_key=self.pose_condition_key,
        )
        joint_out = self._forward_single_branch(
            eps_model=self.joint_eps_model,
            x0=joint_x0,
            ts=ts_joint,
            data=data,
            start_key=self.joint_start_key,
            obser_key=self.joint_obser_key,
            condition_key=self.joint_condition_key,
        )

        pred_pose_x0 = pose_out["pred_x0"]
        pred_joint_x0 = joint_out["pred_x0"]

        if self.detach_pose_for_consistency:
            pred_pose_x0 = pred_pose_x0.detach()
        if self.detach_joint_for_consistency:
            pred_joint_x0 = pred_joint_x0.detach()

        pred_joint_fk_input = self._joint_to_fk_input(pred_joint_x0)
        pred_pose_from_joint = self._run_fk(pred_joint_fk_input)
        pred_pose_from_joint = self._fk_to_reference_repr(pred_pose_from_joint, pred_pose_x0)

        consist_mask = self._build_consistency_mask(pred_pose_x0, data)
        consistency_loss = self._masked_loss(pred_pose_x0, pred_pose_from_joint, consist_mask)

        pose_diff_loss = pose_out["diff_loss"]
        joint_diff_loss = joint_out["diff_loss"]
        pose_lb_loss = self._load_balance_loss(pose_out.get("route", None))
        joint_lb_loss = self._load_balance_loss(joint_out.get("route", None))
        total_loss = (
            self.pose_diff_weight * pose_diff_loss
            + self.joint_diff_weight * joint_diff_loss
            + self.consistency_weight * consistency_loss
            + self.pose_lb_weight * pose_lb_loss
            + self.joint_lb_weight * joint_lb_loss
        )

        return {
            "loss": total_loss,
            "pose_diff_loss": pose_diff_loss,
            "joint_diff_loss": joint_diff_loss,
            "consistency_loss": consistency_loss,
            "pose_lb_loss": pose_lb_loss,
            "joint_lb_loss": joint_lb_loss,
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.lr)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        losses = self(batch)
        self.log("train/loss", losses["loss"], prog_bar=True)
        self.log("train/loss_pose_diff", losses["pose_diff_loss"])
        self.log("train/loss_joint_diff", losses["joint_diff_loss"])
        self.log("train/loss_consistency", losses["consistency_loss"])
        self.log("train/loss_pose_lb", losses["pose_lb_loss"])
        self.log("train/loss_joint_lb", losses["joint_lb_loss"])
        return losses["loss"]

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Optional[torch.Tensor]:
        return None
