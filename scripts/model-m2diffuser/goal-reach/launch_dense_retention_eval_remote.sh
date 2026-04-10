#!/bin/bash
set -euo pipefail

REPO_DIR="/home/lht/01diffuser-acceleration"
DATASET_DIR="/home/lht/03diffuser-generation/dataset/goal-reach-32demo"
LOG_FILE="${REPO_DIR}/experiment_logs/m2/dense_retention_eval.log"
SESSION_NAME="dense_retention_eval"

mkdir -p "${REPO_DIR}/experiment_logs/m2"

screen -dmS "${SESSION_NAME}" bash -lc "\
export PATH=/home/lht/anaconda3/bin:\$PATH && \
eval \"\$(/home/lht/anaconda3/condabin/conda shell.bash hook)\" && \
conda activate m2diffuser && \
cd ${REPO_DIR} && \
{ \
  echo '[eval] R024 start '\"\$(date)\"; \
  DEVICE=cuda:0 DATASET_DIR=${DATASET_DIR} OUTPUT_ROOT=${REPO_DIR}/results/retention_eval SAVE_ROOT=${REPO_DIR}/results/retention_eval MAX_ITEMS=64 GUIDANCE_MODE=off TRAJ_SCALE=1.0 \
    bash scripts/model-m2diffuser/goal-reach/run_retention_eval.sh \
      /home/lht/01diffuser-acceleration/checkpoints/M2-R024-CoKin-ConsDetachJoint/2026-04-09-16-23-05 \
      R024_dense_scale1p0_n64 \
      cokin \
      cokin_pose_mk \
      cokin_joint_mk; \
  echo '[eval] R025 start '\"\$(date)\"; \
  DEVICE=cuda:0 DATASET_DIR=${DATASET_DIR} OUTPUT_ROOT=${REPO_DIR}/results/retention_eval SAVE_ROOT=${REPO_DIR}/results/retention_eval MAX_ITEMS=64 GUIDANCE_MODE=off TRAJ_SCALE=0.7 \
    bash scripts/model-m2diffuser/goal-reach/run_retention_eval.sh \
      /home/lht/01diffuser-acceleration/checkpoints/M2-R025-CoKin-Continue-R024/2026-04-09-19-24-37 \
      R025_dense_scale0p7_n64 \
      cokin \
      cokin_pose_mk \
      cokin_joint_mk; \
  echo '[eval] done '\"\$(date)\"; \
} > ${LOG_FILE} 2>&1"

screen -ls

