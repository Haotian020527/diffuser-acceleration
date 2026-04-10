#!/usr/bin/env bash
set -euo pipefail

CKPT_PATH="${1:-/home/lht/01diffuser-acceleration/checkpoints/M2-R025-CoKin-Continue-R024/2026-04-09-19-24-37}"
TRAJ_SCALE="${TRAJ_SCALE:-0.7}"
MAX_EVAL_ITEMS="${MAX_EVAL_ITEMS:--1}"

python inference_m2diffuser.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir="${CKPT_PATH}" \
  task=mk_m2diffuser_goal_reach \
  task.environment.sim_gui=false \
  task.environment.viz=false \
  +max_eval_items="${MAX_EVAL_ITEMS}" \
  diffuser=cokin \
  diffuser.timesteps=50 \
  diffuser.consistency_weight=1.0 \
  diffuser.detach_joint_for_consistency=true \
  diffuser.sample.converage.optimization=false \
  diffuser.sample.converage.planning=false \
  diffuser.sample.converage.ksteps=1 \
  diffuser.sample.fine_tune.optimization=false \
  diffuser.sample.fine_tune.planning=false \
  diffuser.sample.fine_tune.timesteps=50 \
  diffuser.sample.fine_tune.ksteps=1 \
  diffuser.sample.traj_scale="${TRAJ_SCALE}" \
  model=m2diffuser_mk \
  +model@pose_model=cokin_pose_mk \
  +model@joint_model=cokin_joint_mk \
  model.use_position_embedding=true
