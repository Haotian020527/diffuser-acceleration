import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from eval.metrics import Evaluator
from models.base import create_model
from utils.meckinova_utils import transform_trajectory_torch
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
    if isinstance(obj, dict):
        return {str(k): _to_python(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_to_python(v) for v in obj]
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def _collect_supergroup(groups: Dict[str, Dict[str, Any]]) -> Dict[str, List[Any]]:
    supergroup: Dict[str, List[Any]] = {}
    keys = set()
    for group in groups.values():
        keys.update(group.keys())
    for key in keys:
        merged: List[Any] = []
        for group in groups.values():
            merged.extend(group.get(key, []))
        supergroup[key] = merged
    return supergroup


def _run_single(
    name: str,
    overrides: List[str],
    config_dir: Path,
    max_items: int,
    device: str,
    clamp_before_unnorm: bool,
    skip_world_transform: bool,
    guidance_mode: str,
    guidance_finetune_timesteps: int,
    off_zero_finetune: bool,
    traj_scale: float,
) -> Dict[str, Any]:
    cfg = _compose_cfg(config_dir, overrides)
    _apply_guidance_mode(
        cfg=cfg,
        guidance_mode=guidance_mode,
        guidance_finetune_timesteps=guidance_finetune_timesteps,
        off_zero_finetune=off_zero_finetune,
    )
    ckpt_path = os.path.join(str(cfg.exp_dir), "last.ckpt")

    dm = create_datamodule(cfg=cfg.task.datamodule, slurm=False)
    dl = dm.get_test_dataloader()

    mdl = create_model(cfg, ckpt_path=ckpt_path, slurm=False, device=device)
    mdl = mdl.to(device=device)
    mdl.eval()

    env_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.environment, resolve=False))
    env_cfg.eval = True
    env_cfg.sim_gui = False
    env_cfg.viz = False
    env_cfg.save = False
    env = create_enviroment(env_cfg)

    evaluated = 0
    with torch.no_grad():
        for i, data in enumerate(dl):
            if max_items > 0 and evaluated >= max_items:
                break
            for key in list(data.keys()):
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            start_time = time.time()
            outputs = mdl.sample(data, k=1)
            traj_norm_a = outputs[:, -1, -1, :, :]
            if clamp_before_unnorm:
                traj_norm_a = torch.clamp(traj_norm_a, min=-1.0, max=1.0)
            if traj_scale != 1.0:
                traj_norm_a = traj_norm_a * traj_scale
            traj_unorm_a = MecKinova.unnormalize_joints(traj_norm_a)
            if not skip_world_transform:
                traj_unorm_a = transform_trajectory_torch(
                    traj_unorm_a, torch.inverse(data["trans_mat"]), -data["rot_angle"]
                )
            traj_unorm_a = traj_unorm_a.squeeze(0).detach().cpu().numpy()

            env.evaluate(
                id=i,
                dt=0.08,
                time=time.time() - start_time,
                data=data,
                traj=traj_unorm_a,
                agent_object=MecKinova,
            )
            evaluated += 1

    groups = env.eval.groups
    group_metrics = {k: Evaluator.metrics(v) for k, v in groups.items()}
    supergroup = _collect_supergroup(groups)
    overall = Evaluator.metrics(supergroup)
    diffuser = getattr(mdl, "diffuser", mdl)

    return {
        "name": name,
        "ckpt": ckpt_path,
        "max_items": max_items,
        "evaluated_items": evaluated,
        "clamp_before_unnorm": clamp_before_unnorm,
        "skip_world_transform": skip_world_transform,
        "guidance_mode": guidance_mode,
        "traj_scale": float(traj_scale),
        "effective_guidance": {
            "converage_opt": _to_python(getattr(diffuser, "converage_opt", None)),
            "converage_plan": _to_python(getattr(diffuser, "converage_plan", None)),
            "converage_ksteps": _to_python(getattr(diffuser, "converage_ksteps", None)),
            "fine_tune_opt": _to_python(getattr(diffuser, "fine_tune_opt", None)),
            "fine_tune_plan": _to_python(getattr(diffuser, "fine_tune_plan", None)),
            "fine_tune_timesteps": _to_python(getattr(diffuser, "fine_tune_timesteps", None)),
            "fine_tune_ksteps": _to_python(getattr(diffuser, "fine_tune_ksteps", None)),
        },
        "overall_metrics": _to_python(overall),
        "group_metrics": _to_python(group_metrics),
    }


def _default_jobs(
    dataset_dir: str,
    include_r021: bool,
    include_r022: bool,
    include_r023: bool,
    include_r024: bool,
    include_r025: bool,
) -> List[Tuple[str, List[str]]]:
    jobs: List[Tuple[str, List[str]]] = [
        (
            "DDPM_ref",
            [
                "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/MK-M2Diffuser-Goal-Reach/2024-07-14-09-38-10",
                "task=mk_m2diffuser_goal_reach",
                f"task.datamodule.data_dir={dataset_dir}",
                "diffuser=ddpm",
                "diffuser.timesteps=50",
                "model=m2diffuser_mk",
                "model.use_position_embedding=true",
                "gpus=[0]",
                "no_logging=true",
                "no_checkpointing=true",
            ],
        ),
        (
            "CoKin_R004_Dense",
            [
                "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M1-R004-DenseCoKin/2026-04-09-00-45-57",
                "task=mk_m2diffuser_goal_reach",
                f"task.datamodule.data_dir={dataset_dir}",
                "diffuser=cokin",
                "diffuser.timesteps=50",
                "model=m2diffuser_mk",
                "+model@pose_model=cokin_pose_mk",
                "+model@joint_model=cokin_joint_mk",
                "model.use_position_embedding=true",
                "gpus=[0]",
                "no_logging=true",
                "no_checkpointing=true",
            ],
        ),
    ]
    if include_r021:
        jobs.append(
            (
                "CoKin_R021_NoCons_old",
                [
                    "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M2-R021-CoKin-NoCons/diag-20260409-1518",
                    "task=mk_m2diffuser_goal_reach",
                    f"task.datamodule.data_dir={dataset_dir}",
                    "diffuser=cokin",
                    "diffuser.timesteps=50",
                    "diffuser.consistency_weight=0.0",
                    "model=m2diffuser_mk",
                    "+model@pose_model=cokin_pose_mk",
                    "+model@joint_model=cokin_joint_mk",
                    "model.use_position_embedding=true",
                    "gpus=[0]",
                    "no_logging=true",
                    "no_checkpointing=true",
                ],
            )
        )
    if include_r023:
        jobs.append(
            (
                "CoKin_R023_NoCons_patch",
                [
                    "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M2-R023-CoKin-NoCons-Patch/2026-04-09-16-01-19",
                    "task=mk_m2diffuser_goal_reach",
                    f"task.datamodule.data_dir={dataset_dir}",
                    "diffuser=cokin",
                    "diffuser.timesteps=50",
                    "diffuser.consistency_weight=0.0",
                    "model=m2diffuser_mk",
                    "+model@pose_model=cokin_pose_mk",
                    "+model@joint_model=cokin_joint_mk",
                    "model.use_position_embedding=true",
                    "gpus=[0]",
                    "no_logging=true",
                    "no_checkpointing=true",
                ],
            )
        )
    if include_r024:
        jobs.append(
            (
                "CoKin_R024_ConsDetachJoint",
                [
                    "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M2-R024-CoKin-ConsDetachJoint/2026-04-09-16-23-05",
                    "task=mk_m2diffuser_goal_reach",
                    f"task.datamodule.data_dir={dataset_dir}",
                    "diffuser=cokin",
                    "diffuser.timesteps=50",
                    "diffuser.consistency_weight=1.0",
                    "diffuser.detach_joint_for_consistency=true",
                    "model=m2diffuser_mk",
                    "+model@pose_model=cokin_pose_mk",
                    "+model@joint_model=cokin_joint_mk",
                    "model.use_position_embedding=true",
                    "gpus=[0]",
                    "no_logging=true",
                    "no_checkpointing=true",
                ],
            )
        )
    if include_r025:
        jobs.append(
            (
                "CoKin_R025_ContinueR024",
                [
                    "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M2-R025-CoKin-Continue-R024/2026-04-09-19-24-37",
                    "task=mk_m2diffuser_goal_reach",
                    f"task.datamodule.data_dir={dataset_dir}",
                    "diffuser=cokin",
                    "diffuser.timesteps=50",
                    "diffuser.consistency_weight=1.0",
                    "diffuser.detach_joint_for_consistency=true",
                    "model=m2diffuser_mk",
                    "+model@pose_model=cokin_pose_mk",
                    "+model@joint_model=cokin_joint_mk",
                    "model.use_position_embedding=true",
                    "gpus=[0]",
                    "no_logging=true",
                    "no_checkpointing=true",
                ],
            )
        )
    if include_r022:
        jobs.append(
            (
                "DDPM_R022_32demo",
                [
                    "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M2-R022-DDPM-32demo/diag-20260409-1531",
                    "task=mk_m2diffuser_goal_reach",
                    f"task.datamodule.data_dir={dataset_dir}",
                    "diffuser=ddpm",
                    "diffuser.timesteps=50",
                    "model=m2diffuser_mk",
                    "model.use_position_embedding=true",
                    "gpus=[0]",
                    "no_logging=true",
                    "no_checkpointing=true",
                ],
            )
        )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-side physical feasibility diagnostics")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/lht/03diffuser-generation/dataset/goal-reach-32demo",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=16,
        help="Maximum number of test items to evaluate for each model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device for inference.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_logs/m2/model_side_diag_physical_eval.json",
        help="Path to save JSON diagnostics.",
    )
    parser.add_argument(
        "--clamp-before-unnorm",
        action="store_true",
        help="Clamp sampled normalized joints into [-1,1] before unnormalization.",
    )
    parser.add_argument(
        "--skip-world-transform",
        action="store_true",
        help="Skip transform_trajectory_torch world-frame transform.",
    )
    parser.add_argument(
        "--guidance-mode",
        type=str,
        choices=["default", "on", "off"],
        default="default",
        help=(
            "Control optimization/planning guidance during sampling. "
            "default: keep each diffuser config; on: force all enabled; off: force all disabled."
        ),
    )
    parser.add_argument(
        "--guidance-finetune-timesteps",
        type=int,
        default=50,
        help="Fine-tune timesteps to use when guidance-mode=on.",
    )
    parser.add_argument(
        "--off-zero-finetune",
        action="store_true",
        help=(
            "When guidance-mode=off, also force fine_tune.timesteps=0. "
            "Default keeps existing fine_tune timesteps and only disables optimization/planning gradients."
        ),
    )
    parser.add_argument(
        "--traj-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to sampled normalized joints before unnormalization.",
    )
    parser.add_argument(
        "--include-r021",
        action="store_true",
        help="Also evaluate R021 (old no-consistency short run).",
    )
    parser.add_argument(
        "--include-r022",
        action="store_true",
        help="Also evaluate R022 (DDPM 32-demo short run).",
    )
    parser.add_argument(
        "--include-r023",
        action="store_true",
        help="Also evaluate R023 (patched CoKin no-consistency run).",
    )
    parser.add_argument(
        "--include-r024",
        action="store_true",
        help="Also evaluate R024 (patched CoKin + consistency detach-joint run).",
    )
    parser.add_argument(
        "--include-r025",
        action="store_true",
        help="Also evaluate R025 (R024 continue-training run).",
    )
    parser.add_argument(
        "--only-names",
        type=str,
        default="",
        help=(
            "Optional comma-separated name filters. "
            "Only jobs whose names contain any token will be executed."
        ),
    )
    args = parser.parse_args()

    _register_eval_resolver()
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

    config_dir = REPO_ROOT / "configs"
    jobs = _default_jobs(
        dataset_dir=args.dataset_dir,
        include_r021=args.include_r021,
        include_r022=args.include_r022,
        include_r023=args.include_r023,
        include_r024=args.include_r024,
        include_r025=args.include_r025,
    )
    if args.only_names.strip():
        tokens = [x.strip() for x in args.only_names.split(",") if x.strip()]
        jobs = [job for job in jobs if any(t in job[0] for t in tokens)]
        if not jobs:
            raise ValueError(f"No jobs matched --only-names={tokens}")

    results: List[Dict[str, Any]] = []
    for name, overrides in jobs:
        print(f"[phys-diag] running {name}", flush=True)
        result = _run_single(
            name=name,
            overrides=overrides,
            config_dir=config_dir,
            max_items=args.max_items,
            device=args.device,
            clamp_before_unnorm=args.clamp_before_unnorm,
            skip_world_transform=args.skip_world_transform,
            guidance_mode=args.guidance_mode,
            guidance_finetune_timesteps=args.guidance_finetune_timesteps,
            off_zero_finetune=args.off_zero_finetune,
            traj_scale=args.traj_scale,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_python(results), indent=2), encoding="utf-8")
    print(f"[phys-diag] saved -> {output_path}")


if __name__ == "__main__":
    main()
