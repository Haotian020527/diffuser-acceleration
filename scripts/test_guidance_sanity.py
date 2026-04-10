import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

# Ensure project root is importable when running as:
# `python scripts/test_guidance_sanity.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datamodule.base import create_datamodule
from models.base import create_planner

OmegaConf.register_new_resolver("eval", eval, replace=True)


def move_tensors_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def build_datamodule_cfg(task_cfg):
    dm_cfg = OmegaConf.create(OmegaConf.to_container(task_cfg.datamodule, resolve=False))
    dm_cfg.data_dir = f"/home/lht/03diffuser-generation/dataset/{task_cfg.type}"
    dm_cfg.scene_model_name = "PointTransformer"
    dm_cfg.train_batch_size = task_cfg.train.batch_size
    dm_cfg.val_batch_size = task_cfg.train.batch_size
    dm_cfg.test_batch_size = task_cfg.test.batch_size

    dm_cfg.dataset.task_type = task_cfg.type
    dm_cfg.dataset.train_data_type = task_cfg.train.data_type
    dm_cfg.dataset.val_data_type = task_cfg.train.data_type
    dm_cfg.dataset.test_data_type = task_cfg.test.data_type
    dm_cfg.dataset.num_scene_points = dm_cfg.num_scene_points
    dm_cfg.dataset.num_agent_points = dm_cfg.num_agent_points
    dm_cfg.dataset.num_object_points = dm_cfg.num_object_points
    dm_cfg.dataset.num_placement_area_points = dm_cfg.num_placement_area_points
    dm_cfg.dataset.num_target_points = dm_cfg.num_target_points
    return dm_cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Guidance sanity check for MK planner.")
    parser.add_argument("--task-config", type=str, default="configs/task/mk_m2diffuser_goal_reach.yaml")
    parser.add_argument("--planner-config", type=str, default="configs/planner/mk_motion_policy_planning.yaml")
    parser.add_argument("--output", type=str, default="refine-logs/R002_guidance_sanity.json")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--variance", type=float, default=0.01)
    parser.add_argument("--energy-type", type=str, default="last_frame")
    parser.add_argument("--energy-method", type=str, default="chamfer_distance")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(2026)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2026)

    task_cfg = OmegaConf.load(args.task_config)
    dm_cfg = build_datamodule_cfg(task_cfg)
    dm_cfg.test_batch_size = 1
    dm_cfg.num_workers = 0

    planner_cfg = OmegaConf.load(args.planner_config)
    planner_cfg.goal_reach_energy = True
    planner_cfg.goal_reach_energy_type = args.energy_type
    planner_cfg.goal_reach_energy_method = args.energy_method
    planner_cfg.grasp_energy = False
    planner_cfg.place_energy = False
    planner_cfg.scale_type = "div_var"

    dm = create_datamodule(cfg=dm_cfg, slurm=False)
    dl = dm.get_test_dataloader()
    batch = next(iter(dl))
    batch = move_tensors_to_device(batch, device)

    if "x" not in batch:
        raise KeyError("Batch missing key `x` for trajectory tensor.")
    if "target_pc_a" not in batch:
        raise KeyError("Batch missing key `target_pc_a`; goal-reach guidance cannot run.")

    planner = create_planner(planner_cfg, device=str(device))

    x = batch["x"].clone().detach()
    variance = torch.full_like(x, fill_value=args.variance, device=device)

    start = time.time()
    grad = planner.gradient(x, batch, variance)
    elapsed = time.time() - start

    finite = torch.isfinite(grad).all().item()
    grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=1)
    mean_norm = grad_norm.mean().item()
    max_abs = grad.abs().max().item()

    result = {
        "run_id": "R002",
        "status": "PASSED" if finite and mean_norm > 0 else "FAILED",
        "device": str(device),
        "elapsed_sec": elapsed,
        "grad_finite": bool(finite),
        "grad_mean_norm": float(mean_norm),
        "grad_max_abs": float(max_abs),
        "variance": float(args.variance),
        "energy_type": args.energy_type,
        "energy_method": args.energy_method,
        "notes": "Gradient sanity checks planner guidance with dataset batch and target point cloud.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if result["status"] != "PASSED":
        raise RuntimeError(f"Guidance sanity failed: {result}")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
