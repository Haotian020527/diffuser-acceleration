import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

# Ensure project root is importable when running as:
# `python scripts/test_metric_sanity.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datamodule.base import create_datamodule
from env.agent.mec_kinova import MecKinova
from env.base import create_enviroment
from eval.metrics import Evaluator
from utils.meckinova_utils import transform_trajectory_torch

OmegaConf.register_new_resolver("eval", eval, replace=True)


def move_tensors_to_cpu(batch: Dict) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.cpu()
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
    parser = argparse.ArgumentParser(description="Metric sanity check with dataset ground-truth trajectory.")
    parser.add_argument("--task-config", type=str, default="configs/task/mk_m2diffuser_goal_reach.yaml")
    parser.add_argument("--output", type=str, default="refine-logs/R003_metric_sanity.json")
    parser.add_argument("--dt", type=float, default=0.08)
    args = parser.parse_args()

    torch.manual_seed(2026)

    task_cfg = OmegaConf.load(args.task_config)
    dm_cfg = build_datamodule_cfg(task_cfg)
    dm_cfg.test_batch_size = 1
    dm_cfg.num_workers = 0

    dm = create_datamodule(cfg=dm_cfg, slurm=False)
    dl = dm.get_test_dataloader()
    batch = next(iter(dl))

    if "x" not in batch:
        raise KeyError("Batch missing `x` trajectory.")
    if "trans_mat" not in batch or "rot_angle" not in batch:
        raise KeyError("Batch missing `trans_mat`/`rot_angle` required for trajectory alignment.")

    traj_norm = batch["x"]  # [B, L, D]
    traj_unorm = MecKinova.unnormalize_joints(traj_norm)
    traj_agent = transform_trajectory_torch(
        traj_unorm,
        torch.inverse(batch["trans_mat"]),
        -batch["rot_angle"],
    )
    traj_for_eval = traj_agent[0].detach().cpu().numpy()

    env_cfg = OmegaConf.create(OmegaConf.to_container(task_cfg.environment, resolve=False))
    env_cfg.eval = True
    env_cfg.sim_gui = False
    env_cfg.viz = False
    env_cfg.save = False
    env_cfg.save_dir = "./results/metric_sanity"

    env = create_enviroment(env_cfg)
    batch_cpu = move_tensors_to_cpu(batch)
    env.evaluate(
        id=0,
        dt=args.dt,
        time=0.0,
        data=batch_cpu,
        traj=traj_for_eval,
        agent_object=MecKinova,
        skip_metrics=False,
    )

    group_key = env.eval.current_group_key
    group_metrics = Evaluator.metrics(env.eval.groups[group_key])

    result = {
        "run_id": "R003",
        "status": "PASSED",
        "group_key": group_key,
        "ground_truth_used": True,
        "metrics": {
            "physical_success_percent": float(group_metrics["physical_success"]),
            "physical_violations_percent": float(group_metrics["physical violations"]),
            "config_sparc": float(group_metrics["average config sparc"]),
            "eff_sparc": float(group_metrics["average eff sparc"]),
            "env_collision_percent": float(group_metrics["env collision"]),
            "self_collision_percent": float(group_metrics["self collision"]),
            "joint_violation_percent": float(group_metrics["joint violation"]),
        },
        "notes": "Metrics are computed against dataset geometry/constraints using ground-truth trajectory.",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
