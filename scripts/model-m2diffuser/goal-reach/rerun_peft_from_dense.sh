#!/usr/bin/env bash
set -euo pipefail

DEFAULT_BASE_CKPT="/home/lht/01diffuser-acceleration/checkpoints/M1-R004-DenseCoKin/2026-04-09-00-45-57/last.ckpt"
BASE_CKPT="${1:-${DEFAULT_BASE_CKPT}}"
NUM_GPUS="${2:-1}"
MODES_CSV="${3:-lora_cokin,adapter_cokin,fadapter_cokin}"
shift $(( $# >= 3 ? 3 : $# ))
EXTRA_ARGS=("$@")

IFS=',' read -r -a MODES <<< "${MODES_CSV}"

case "${BASE_CKPT}" in
    *M1-R004-DenseCoKin*|*M2-R024-CoKin-ConsDetachJoint*|*M2-R025-CoKin-Continue-R024*)
        echo "Warning: ${BASE_CKPT} looks like a few-shot or diagnostic checkpoint, not the original pretrained dense CoKin base."
        echo "Warning: This is valid for engineering comparison, but it weakens the proposal-aligned pretrained-adaptation claim."
        ;;
esac

echo "Warm-start base: ${BASE_CKPT}"
echo "GPUs: ${NUM_GPUS}"
echo "Modes: ${MODES_CSV}"
if [ "${BASE_CKPT}" = "${DEFAULT_BASE_CKPT}" ]; then
    echo "Note: using inferred default dense CoKin base from current experiment context."
fi

for mode in "${MODES[@]}"; do
    if [ -z "${mode}" ]; then
        continue
    fi
    echo "==== Launching ${mode} from ${BASE_CKPT} ===="
    INIT_FROM_CKPT="${BASE_CKPT}" \
        ./scripts/model-m2diffuser/goal-reach/train.sh \
        "${NUM_GPUS}" \
        "${mode}" \
        none \
        "${EXTRA_ARGS[@]}"
done
