import argparse
import json
import os
import random
import sys
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
from models.base import create_model
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


def _tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    flat = x.float().reshape(-1)
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_gt_1_ratio": float((flat.abs() > 1.0).float().mean().item()),
        "abs_gt_1p2_ratio": float((flat.abs() > 1.2).float().mean().item()),
    }


def _run_single(
    name: str,
    overrides: List[str],
    config_dir: Path,
    n_items: int,
    device: str,
) -> Dict[str, Any]:
    no_ckpt = "__no_ckpt__" in overrides
    clean_overrides = [x for x in overrides if x != "__no_ckpt__"]
    cfg = _compose_cfg(config_dir, clean_overrides)
    ckpt_path = None if no_ckpt else os.path.join(str(cfg.exp_dir), "last.ckpt")

    dm = create_datamodule(cfg=cfg.task.datamodule, slurm=False)
    dl = dm.get_test_dataloader()

    mdl = create_model(cfg, ckpt_path=ckpt_path, slurm=False, device=device)
    mdl = mdl.to(device=device)
    mdl.eval()

    gt_all: List[torch.Tensor] = []
    x0_all: List[torch.Tensor] = []
    xT_all: List[torch.Tensor] = []
    start_lock_mse: List[float] = []

    with torch.no_grad():
        for i, data in enumerate(dl):
            if i >= n_items:
                break
            for key, value in list(data.items()):
                if torch.is_tensor(value):
                    data[key] = value.to(device)

            chain = mdl.p_sample_loop(data)
            xT = chain[:, 0]
            x0 = chain[:, -1]
            gt = data["x"]

            gt_all.append(gt.detach().cpu())
            x0_all.append(x0.detach().cpu())
            xT_all.append(xT.detach().cpu())

            if "start" in data:
                start = data["start"]
                start_len = start.shape[1]
                start_lock_mse.append(float(((x0[:, :start_len, :] - start) ** 2).mean().item()))

    gt_cat = torch.cat(gt_all, dim=0)
    x0_cat = torch.cat(x0_all, dim=0)
    xT_cat = torch.cat(xT_all, dim=0)

    return {
        "name": name,
        "ckpt": ckpt_path if ckpt_path is not None else "<random_init>",
        "n_items": int(gt_cat.shape[0]),
        "shape": {
            "gt": list(gt_cat.shape),
            "x0": list(x0_cat.shape),
            "xT": list(xT_cat.shape),
        },
        "gt_stats": _tensor_stats(gt_cat),
        "xT_stats": _tensor_stats(xT_cat),
        "x0_stats": _tensor_stats(x0_cat),
        "denoise_std_ratio_x0_over_xT": float(
            (x0_cat.std(unbiased=False) / (xT_cat.std(unbiased=False) + 1e-8)).item()
        ),
        "mse_x0_vs_gt": float(((x0_cat - gt_cat) ** 2).mean().item()),
        "start_lock_mse_mean": float(np.mean(start_lock_mse)) if start_lock_mse else None,
    }


def _default_jobs(
    dataset_dir: str,
    include_random_cokin: bool,
    include_r021_nocons: bool,
    include_r022_ddpm32demo: bool,
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
        (
            "CoKin_R007_FAdapter",
            [
                "exp_dir=/home/lht/01diffuser-acceleration/checkpoints/M2-R007-CoKin-FAdapter/2026-04-09-10-36-28",
                "task=mk_m2diffuser_goal_reach",
                f"task.datamodule.data_dir={dataset_dir}",
                "diffuser=cokin",
                "diffuser.timesteps=50",
                "model=m2diffuser_mk",
                "+model@pose_model=fadapter_pose_mk",
                "+model@joint_model=fadapter_joint_mk",
                "model.use_position_embedding=true",
                "gpus=[0]",
                "no_logging=true",
                "no_checkpointing=true",
            ],
        ),
    ]
    if include_random_cokin:
        jobs.append(
            (
                "CoKin_RandomInit",
                [
                    "__no_ckpt__",
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
            )
        )
    if include_r021_nocons:
        jobs.append(
            (
                "CoKin_R021_NoCons",
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
    if include_r022_ddpm32demo:
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
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Model-side sampling distribution diagnostics")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/lht/03diffuser-generation/dataset/goal-reach-32demo",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=16,
        help="Number of test items to sample for each model.",
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
        default="experiment_logs/m2/model_side_diag_sampling_stats.json",
        help="Path to save JSON diagnostics.",
    )
    parser.add_argument(
        "--include-random-cokin",
        action="store_true",
        help="Also run a random-initialized CoKin baseline (no checkpoint load).",
    )
    parser.add_argument(
        "--include-r021-nocons",
        action="store_true",
        help="Also run the R021 short training checkpoint with consistency_weight=0.",
    )
    parser.add_argument(
        "--include-r022-ddpm32demo",
        action="store_true",
        help="Also run the R022 short DDPM checkpoint trained from scratch on 32-demo.",
    )
    parser.add_argument(
        "--include-r024",
        action="store_true",
        help="Also run the R024 checkpoint.",
    )
    parser.add_argument(
        "--include-r025",
        action="store_true",
        help="Also run the R025 checkpoint.",
    )
    args = parser.parse_args()

    _register_eval_resolver()
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

    repo_root = REPO_ROOT
    config_dir = repo_root / "configs"
    jobs = _default_jobs(
        dataset_dir=args.dataset_dir,
        include_random_cokin=args.include_random_cokin,
        include_r021_nocons=args.include_r021_nocons,
        include_r022_ddpm32demo=args.include_r022_ddpm32demo,
        include_r024=args.include_r024,
        include_r025=args.include_r025,
    )

    results: List[Dict[str, Any]] = []
    for name, overrides in jobs:
        print(f"[diag] running {name}", flush=True)
        result = _run_single(
            name=name,
            overrides=overrides,
            config_dir=config_dir,
            n_items=args.n_items,
            device=args.device,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[diag] saved -> {output_path}")


if __name__ == "__main__":
    main()
