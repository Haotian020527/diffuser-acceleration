import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datamodule.base import create_datamodule
from env.agent.mec_kinova import MecKinova
from env.base import create_enviroment
from env.sampler.mk_sampler import MecKinovaSampler
from eval.metrics import Evaluator
from models.base import create_model
from utils.meckinova_utils import transform_trajectory_numpy, transform_trajectory_torch
from utils.misc import compute_model_dim


def _register_eval_resolver() -> None:
    try:
        OmegaConf.register_new_resolver("eval", eval, replace=True)
    except Exception:
        pass


def _compose_cfg(config_dir: Path, overrides: List[str]):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="default", overrides=overrides)
    cfg.model.d_x = compute_model_dim(cfg.task)
    return cfg


def _ensure_sample_cfg(cfg) -> Any:
    if "sample" not in cfg.diffuser or cfg.diffuser.sample is None:
        cfg.diffuser.sample = OmegaConf.create({})
    sample_cfg = cfg.diffuser.sample
    if "converage" not in sample_cfg or sample_cfg.converage is None:
        sample_cfg.converage = OmegaConf.create({})
    if "fine_tune" not in sample_cfg or sample_cfg.fine_tune is None:
        sample_cfg.fine_tune = OmegaConf.create({})
    return sample_cfg


def _apply_guidance_mode(
    cfg,
    guidance_mode: str,
    guidance_finetune_timesteps: int,
    off_zero_finetune: bool,
) -> None:
    if guidance_mode == "default":
        return

    sample_cfg = _ensure_sample_cfg(cfg)
    enable = guidance_mode == "on"

    sample_cfg.converage.optimization = bool(enable)
    sample_cfg.converage.planning = bool(enable)
    sample_cfg.converage.ksteps = int(sample_cfg.converage.get("ksteps", 1))

    sample_cfg.fine_tune.optimization = bool(enable)
    sample_cfg.fine_tune.planning = bool(enable)
    sample_cfg.fine_tune.ksteps = int(sample_cfg.fine_tune.get("ksteps", 1))
    if enable:
        sample_cfg.fine_tune.timesteps = int(guidance_finetune_timesteps)
    elif off_zero_finetune:
        sample_cfg.fine_tune.timesteps = 0


def _to_python(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_python(v) for v in obj]
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return _to_python(obj.item())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return _to_python(obj.item())
        return obj.detach().cpu().tolist()
    return obj


def _tensor_to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _avg_pool_lowpass(x: torch.Tensor, low_freq_ratio: float) -> torch.Tensor:
    seq_len = x.shape[-1]
    if seq_len <= 2:
        return x
    kernel = int(round(1.0 / max(low_freq_ratio, 1e-3)))
    kernel = max(3, kernel)
    if kernel % 2 == 0:
        kernel += 1
    kernel = min(kernel, seq_len if seq_len % 2 == 1 else max(3, seq_len - 1))
    pad = kernel // 2
    return torch.nn.functional.avg_pool1d(
        x.to(torch.float32), kernel_size=kernel, stride=1, padding=pad
    ).to(dtype=x.dtype)


def _rotation_error_deg(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    rel = pred_rot @ gt_rot.transpose(-1, -2)
    trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    return torch.rad2deg(torch.acos(cos_theta))


def _task_success_from_world_traj(
    sampler: MecKinovaSampler,
    pred_world: torch.Tensor,
    gt_world: torch.Tensor,
    pos_threshold: float,
    rot_threshold_deg: float,
) -> Dict[str, float]:
    pred_pose = sampler.end_effector_pose(pred_world[-1]).squeeze(0)
    gt_pose = sampler.end_effector_pose(gt_world[-1]).squeeze(0)

    pred_xyz = pred_pose[:3, 3]
    gt_xyz = gt_pose[:3, 3]
    pos_error = torch.linalg.norm(pred_xyz - gt_xyz).item()

    pred_rot = pred_pose[:3, :3].unsqueeze(0)
    gt_rot = gt_pose[:3, :3].unsqueeze(0)
    rot_error_deg = float(_rotation_error_deg(pred_rot, gt_rot).item())

    pred_world_neg = pred_world[-1].clone()
    pred_world_neg[-1] += math.pi
    if pred_world_neg[-1] > math.pi:
        pred_world_neg[-1] -= 2 * math.pi
    pred_pose_neg = sampler.end_effector_pose(pred_world_neg).squeeze(0)
    pred_rot_neg = pred_pose_neg[:3, :3].unsqueeze(0)
    pos_error_neg = torch.linalg.norm(pred_pose_neg[:3, 3] - gt_xyz).item()
    rot_error_neg_deg = float(_rotation_error_deg(pred_rot_neg, gt_rot).item())

    if rot_error_deg <= rot_error_neg_deg:
        best_pos_error = pos_error
        best_rot_error_deg = rot_error_deg
    else:
        best_pos_error = pos_error_neg
        best_rot_error_deg = rot_error_neg_deg

    task_success = (
        best_pos_error < pos_threshold and best_rot_error_deg < rot_threshold_deg
    )
    return {
        "task_success": bool(task_success),
        "final_position_error_m": float(best_pos_error),
        "final_orientation_error_deg": float(best_rot_error_deg),
    }


def _fk_drift_metrics(
    sampler: MecKinovaSampler,
    pred_world: torch.Tensor,
    gt_world: torch.Tensor,
) -> Dict[str, float]:
    horizon = min(pred_world.shape[0], gt_world.shape[0])
    pred_fk = sampler.end_effector_pose(pred_world[:horizon])
    gt_fk = sampler.end_effector_pose(gt_world[:horizon])

    pos_error = torch.linalg.norm(pred_fk[:, :3, 3] - gt_fk[:, :3, 3], dim=-1)
    rot_error_deg = _rotation_error_deg(pred_fk[:, :3, :3], gt_fk[:, :3, :3])
    return {
        "fk_position_mae_m": float(pos_error.mean().item()),
        "fk_position_max_m": float(pos_error.max().item()),
        "fk_orientation_mae_deg": float(rot_error_deg.mean().item()),
        "fk_orientation_max_deg": float(rot_error_deg.max().item()),
    }


def _spectral_metrics(
    pred_norm_agent: torch.Tensor,
    gt_norm_agent: torch.Tensor,
    low_freq_ratio: float,
) -> Dict[str, float]:
    pred_ch = pred_norm_agent.transpose(0, 1).unsqueeze(0)
    gt_ch = gt_norm_agent.transpose(0, 1).unsqueeze(0)
    pred_low = _avg_pool_lowpass(pred_ch, low_freq_ratio)
    gt_low = _avg_pool_lowpass(gt_ch, low_freq_ratio)
    pred_high = pred_ch - pred_low
    gt_high = gt_ch - gt_low

    return {
        "low_band_spectral_mse": float(torch.mean((pred_low - gt_low) ** 2).item()),
        "high_band_spectral_mse": float(torch.mean((pred_high - gt_high) ** 2).item()),
        "full_traj_mse": float(torch.mean((pred_ch - gt_ch) ** 2).item()),
    }


def _build_overrides(args: argparse.Namespace) -> List[str]:
    overrides = [
        f"exp_dir={args.exp_dir}",
        f"task={args.task}",
        f"task.datamodule.data_dir={args.dataset_dir}",
        f"diffuser={args.diffuser}",
        f"model={args.model}",
        "model.use_position_embedding=true",
        "gpus=[0]",
        "no_logging=true",
        "no_checkpointing=true",
    ]
    if args.timesteps is not None:
        overrides.append(f"diffuser.timesteps={args.timesteps}")
    if args.pose_model:
        overrides.append(f"+model@pose_model={args.pose_model}")
    if args.joint_model:
        overrides.append(f"+model@joint_model={args.joint_model}")
    overrides.extend(args.override)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a goal-reach checkpoint and save parseable JSON metrics."
    )
    parser.add_argument("--exp-dir", type=str, required=True, help="Experiment directory containing last.ckpt.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/lht/03diffuser-generation/dataset/goal-reach-32demo",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument("--task", type=str, default="mk_m2diffuser_goal_reach")
    parser.add_argument("--diffuser", type=str, default="cokin")
    parser.add_argument("--model", type=str, default="m2diffuser_mk")
    parser.add_argument("--pose-model", type=str, default="")
    parser.add_argument("--joint-model", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--override", action="append", default=[], help="Extra Hydra override. Repeatable.")
    parser.add_argument("--max-items", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="Optional directory to save per-item env results. Leave empty to disable.",
    )
    parser.add_argument(
        "--guidance-mode",
        type=str,
        choices=["default", "on", "off"],
        default="default",
    )
    parser.add_argument("--guidance-finetune-timesteps", type=int, default=50)
    parser.add_argument("--off-zero-finetune", action="store_true")
    parser.add_argument("--clamp-before-unnorm", action="store_true")
    parser.add_argument("--skip-world-transform", action="store_true")
    parser.add_argument("--traj-scale", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.08)
    parser.add_argument("--task-success-pos-threshold", type=float, default=0.04)
    parser.add_argument("--task-success-rot-threshold-deg", type=float, default=20.0)
    args = parser.parse_args()

    _register_eval_resolver()
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

    overrides = _build_overrides(args)
    cfg = _compose_cfg(REPO_ROOT / "configs", overrides)
    _apply_guidance_mode(
        cfg=cfg,
        guidance_mode=args.guidance_mode,
        guidance_finetune_timesteps=args.guidance_finetune_timesteps,
        off_zero_finetune=args.off_zero_finetune,
    )

    dm = create_datamodule(cfg=cfg.task.datamodule, slurm=False)
    dl = dm.get_test_dataloader()

    ckpt_path = os.path.join(str(cfg.exp_dir), "last.ckpt")
    mdl = create_model(cfg, ckpt_path=ckpt_path, slurm=False, device=args.device)
    mdl = mdl.to(device=args.device)
    mdl.eval()

    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.environment, resolve=False))
    env_cfg.eval = True
    env_cfg.sim_gui = False
    env_cfg.viz = False
    env_cfg.save = bool(args.save_dir)
    if args.save_dir:
        env_cfg.save_dir = args.save_dir
    env = create_enviroment(env_cfg)

    sampler = MecKinovaSampler(args.device, num_fixed_points=1024, use_cache=True)
    robot = MecKinova()

    items: List[Dict[str, Any]] = []
    with torch.no_grad():
        for item_idx, data in enumerate(dl):
            if args.max_items > 0 and item_idx >= args.max_items:
                break

            for key in list(data.keys()):
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(args.device)

            start = time.time()
            outputs = mdl.sample(data, k=1)
            traj_norm_agent = outputs[:, -1, -1, :, :].clone()
            if args.clamp_before_unnorm:
                traj_norm_agent = torch.clamp(traj_norm_agent, min=-1.0, max=1.0)
            if args.traj_scale != 1.0:
                traj_norm_agent = traj_norm_agent * args.traj_scale

            traj_eval_agent = MecKinova.unnormalize_joints(traj_norm_agent.clone())
            if not args.skip_world_transform:
                traj_eval_agent = transform_trajectory_torch(
                    traj_eval_agent,
                    torch.inverse(data["trans_mat"]),
                    -data["rot_angle"],
                )

            traj_eval_agent_np = traj_eval_agent.squeeze(0).detach().cpu().numpy()
            env.evaluate(
                id=item_idx,
                dt=args.dt,
                time=time.time() - start,
                data=data,
                traj=traj_eval_agent_np,
                agent_object=MecKinova,
            )
            eval_result = dict(env.eval.current_result) if getattr(env, "eval", None) is not None else {}

            pred_world = transform_trajectory_numpy(
                traj_eval_agent_np,
                _tensor_to_numpy(data["T_aw"].squeeze(0)),
                _tensor_to_numpy(data["agent_init_pos"].squeeze(0))[-1],
            )
            pred_world_t = torch.as_tensor(pred_world, device=args.device, dtype=torch.float32)

            if "traj_w" in data:
                gt_world_t = data["traj_w"].squeeze(0).to(dtype=torch.float32)
            else:
                gt_agent = MecKinova.unnormalize_joints(data["x"].squeeze(0))
                gt_world = transform_trajectory_torch(
                    gt_agent.unsqueeze(0),
                    data["T_aw"],
                    data["agent_init_pos"][:, -1],
                ).squeeze(0)
                gt_world_t = gt_world.to(dtype=torch.float32)

            gt_norm_agent = data["x"].squeeze(0).to(dtype=torch.float32)
            pred_norm_agent_single = traj_norm_agent.squeeze(0).to(dtype=torch.float32)

            task_metrics = _task_success_from_world_traj(
                sampler=sampler,
                pred_world=pred_world_t,
                gt_world=gt_world_t,
                pos_threshold=args.task_success_pos_threshold,
                rot_threshold_deg=args.task_success_rot_threshold_deg,
            )
            fk_metrics = _fk_drift_metrics(
                sampler=sampler,
                pred_world=pred_world_t,
                gt_world=gt_world_t,
            )
            spectral_metrics = _spectral_metrics(
                pred_norm_agent=pred_norm_agent_single,
                gt_norm_agent=gt_norm_agent,
                low_freq_ratio=float(
                    OmegaConf.select(cfg, "joint_model.low_freq_ratio", default=0.25)
                ),
            )

            physical_success = bool(eval_result.get("physical_success", False))
            combined_success = physical_success and task_metrics["task_success"]
            goal_pose = robot.get_eff_pose(_tensor_to_numpy(gt_world_t[-1]))
            pred_pose = robot.get_eff_pose(_tensor_to_numpy(pred_world_t[-1]))

            items.append(
                {
                    "item_id": item_idx,
                    "scene_name": str(data["scene_name"][0]) if "scene_name" in data else None,
                    "task_name": str(data["task_name"][0]) if "task_name" in data else None,
                    "physical_success": physical_success,
                    "combined_success": bool(combined_success),
                    "goal_pose_xyz": _to_python(goal_pose[:3, 3]),
                    "pred_pose_xyz": _to_python(pred_pose[:3, 3]),
                    **_to_python(eval_result),
                    **task_metrics,
                    **fk_metrics,
                    **spectral_metrics,
                }
            )

    group_key = env.eval.current_group_key
    group_metrics = Evaluator.metrics(env.eval.groups[group_key]) if group_key else {}

    def _mean(items_list: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [float(item[key]) for item in items_list if key in item]
        if not values:
            return None
        return float(np.mean(values))

    task_success_percent = 100.0 * sum(
        1 for item in items if item["task_success"]
    ) / max(len(items), 1)
    combined_success_percent = 100.0 * sum(
        1 for item in items if item["combined_success"]
    ) / max(len(items), 1)

    diffuser = getattr(mdl, "diffuser", mdl)
    summary = {
        "exp_dir": str(cfg.exp_dir),
        "ckpt_path": ckpt_path,
        "evaluated_items": len(items),
        "task_success_percent": float(task_success_percent),
        "combined_success_percent": float(combined_success_percent),
        "physical_metrics": _to_python(group_metrics),
        "mean_final_position_error_m": _mean(items, "final_position_error_m"),
        "mean_final_orientation_error_deg": _mean(items, "final_orientation_error_deg"),
        "mean_fk_position_mae_m": _mean(items, "fk_position_mae_m"),
        "mean_fk_orientation_mae_deg": _mean(items, "fk_orientation_mae_deg"),
        "mean_low_band_spectral_mse": _mean(items, "low_band_spectral_mse"),
        "mean_high_band_spectral_mse": _mean(items, "high_band_spectral_mse"),
        "mean_full_traj_mse": _mean(items, "full_traj_mse"),
        "effective_guidance": {
            "converage_opt": _to_python(getattr(diffuser, "converage_opt", None)),
            "converage_plan": _to_python(getattr(diffuser, "converage_plan", None)),
            "converage_ksteps": _to_python(getattr(diffuser, "converage_ksteps", None)),
            "fine_tune_opt": _to_python(getattr(diffuser, "fine_tune_opt", None)),
            "fine_tune_plan": _to_python(getattr(diffuser, "fine_tune_plan", None)),
            "fine_tune_timesteps": _to_python(getattr(diffuser, "fine_tune_timesteps", None)),
            "fine_tune_ksteps": _to_python(getattr(diffuser, "fine_tune_ksteps", None)),
        },
    }

    output = {
        "config": {
            "task": args.task,
            "diffuser": args.diffuser,
            "model": args.model,
            "pose_model": args.pose_model or None,
            "joint_model": args.joint_model or None,
            "dataset_dir": args.dataset_dir,
            "guidance_mode": args.guidance_mode,
            "traj_scale": float(args.traj_scale),
            "clamp_before_unnorm": bool(args.clamp_before_unnorm),
            "skip_world_transform": bool(args.skip_world_transform),
            "save_dir": args.save_dir or None,
            "extra_overrides": args.override,
        },
        "summary": summary,
        "items": items,
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_python(output), indent=2), encoding="utf-8")
    print(json.dumps(_to_python(summary), indent=2))
    print(f"[eval] saved -> {output_path}")


if __name__ == "__main__":
    main()

