#!/usr/bin/env bash
set -euo pipefail

EXP_DIR="${1:-}"
RUN_TAG="${2:-}"
DIFFUSER="${3:-cokin}"
POSE_MODEL="${4:-none}"
JOINT_MODEL="${5:-none}"
shift $(( $# >= 5 ? 5 : $# ))
EXTRA_ARGS=("$@")

if [ -z "${EXP_DIR}" ] || [ -z "${RUN_TAG}" ]; then
    echo "Usage: $0 <exp_dir> <run_tag> [diffuser] [pose_model|none] [joint_model|none] [extra args ...]"
    echo "Example: $0 /abs/path/to/exp R005_warm cokin lora_pose_mk lora_joint_mk --max-items 16"
    exit 1
fi

DATASET_DIR="${DATASET_DIR:-/home/lht/03diffuser-generation/dataset/goal-reach-32demo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/retention_eval}"
SAVE_ROOT="${SAVE_ROOT:-${OUTPUT_ROOT}}"
TIMESTEPS="${TIMESTEPS:-50}"
DEVICE="${DEVICE:-cuda:0}"
GUIDANCE_MODE="${GUIDANCE_MODE:-off}"
TRAJ_SCALE="${TRAJ_SCALE:-1.0}"
MAX_ITEMS="${MAX_ITEMS:-16}"

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
SAVE_DIR="${SAVE_ROOT}/${RUN_TAG}/raw"
OUTPUT_JSON="${OUTPUT_DIR}/summary.json"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${SAVE_DIR}"

ARGS=(
    --exp-dir "${EXP_DIR}"
    --dataset-dir "${DATASET_DIR}"
    --diffuser "${DIFFUSER}"
    --timesteps "${TIMESTEPS}"
    --device "${DEVICE}"
    --guidance-mode "${GUIDANCE_MODE}"
    --traj-scale "${TRAJ_SCALE}"
    --max-items "${MAX_ITEMS}"
    --output "${OUTPUT_JSON}"
    --save-dir "${SAVE_DIR}"
)

if [ "${POSE_MODEL}" != "none" ]; then
    ARGS+=(--pose-model "${POSE_MODEL}")
fi
if [ "${JOINT_MODEL}" != "none" ]; then
    ARGS+=(--joint-model "${JOINT_MODEL}")
fi

echo "[retention-eval] exp_dir=${EXP_DIR}"
echo "[retention-eval] output=${OUTPUT_JSON}"
echo "[retention-eval] save_dir=${SAVE_DIR}"

python scripts/eval_goal_reach_checkpoint.py "${ARGS[@]}" "${EXTRA_ARGS[@]}"

