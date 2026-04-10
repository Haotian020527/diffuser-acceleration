#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_CKPT="/home/lht/01diffuser-acceleration/checkpoints/M1-R004-DenseCoKin/2026-04-09-00-45-57/last.ckpt"
BASE_CKPT="${1:-${DEFAULT_BASE_CKPT}}"
NUM_GPUS="${2:-1}"
shift $(( $# >= 2 ? 2 : $# ))
EXTRA_ARGS=("$@")

echo "Warm-start base: ${BASE_CKPT}"
echo "GPUs: ${NUM_GPUS}"

launch_fadapter_variant() {
    local exp_name="$1"
    shift
    INIT_FROM_CKPT="${BASE_CKPT}" \
        ./scripts/model-m2diffuser/goal-reach/train.sh \
        "${NUM_GPUS}" \
        fadapter_cokin \
        none \
        "exp_name=${exp_name}" \
        "$@" \
        "${EXTRA_ARGS[@]}"
}

launch_fadapter_variant \
    "M3-R008-CoKin-FAdapter-NoAnchor" \
    "+diffuser.low_band_anchor_weight=0.0" \
    "+diffuser.band_leak_weight=0.05" \
    "+diffuser.anchor_pose_weight=1.0" \
    "+diffuser.anchor_joint_weight=0.5"

launch_fadapter_variant \
    "M3-R009-CoKin-FAdapter-NoLeak" \
    "+diffuser.low_band_anchor_weight=0.5" \
    "+diffuser.band_leak_weight=0.0" \
    "+diffuser.anchor_pose_weight=1.0" \
    "+diffuser.anchor_joint_weight=0.5"

launch_fadapter_variant \
    "M3-R010-CoKin-FAdapter-Symmetric" \
    "+diffuser.low_band_anchor_weight=0.5" \
    "+diffuser.band_leak_weight=0.05" \
    "+diffuser.anchor_pose_weight=1.0" \
    "+diffuser.anchor_joint_weight=0.5" \
    "pose_model.low_band_rank=12" \
    "pose_model.high_band_rank=2" \
    "joint_model.low_band_rank=12" \
    "joint_model.high_band_rank=2"

INIT_FROM_CKPT="${BASE_CKPT}" \
    ./scripts/model-m2diffuser/goal-reach/train.sh \
    "${NUM_GPUS}" \
    moe_cokin \
    none \
    "exp_name=M3-R011-CoKin-MoE" \
    "${EXTRA_ARGS[@]}"

