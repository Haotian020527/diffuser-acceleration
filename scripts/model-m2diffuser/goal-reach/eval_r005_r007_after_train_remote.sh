#!/bin/bash
set -euo pipefail

REPO_DIR="/home/lht/01diffuser-acceleration"
TRAIN_SESSION="r005_r007_warm"
DATASET_DIR="/home/lht/03diffuser-generation/dataset/goal-reach-32demo"
EVAL_LOG="${REPO_DIR}/experiment_logs/m2/r005_r007_warm_retention_eval.log"
mkdir -p "${REPO_DIR}/experiment_logs/m2"

wait_for_training_end() {
    while screen -ls | grep -q "${TRAIN_SESSION}"; do
        sleep 120
    done
}

latest_ckpt_dir() {
    local exp_name="$1"
    local base_dir="${REPO_DIR}/checkpoints/${exp_name}"
    find "${base_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

run_eval() {
    local run_tag="$1"
    local exp_name="$2"
    local diffuser="$3"
    local pose_cfg="$4"
    local joint_cfg="$5"
    local ckpt_dir
    ckpt_dir="$(latest_ckpt_dir "${exp_name}")"
    if [ -z "${ckpt_dir}" ]; then
        echo "[eval] missing checkpoint dir for ${run_tag} (${exp_name})" | tee -a "${EVAL_LOG}"
        return 1
    fi

    echo "[eval] ${run_tag} -> ${ckpt_dir}" | tee -a "${EVAL_LOG}"
    OUTPUT_ROOT="${REPO_DIR}/results/retention_eval" \
    SAVE_ROOT="${REPO_DIR}/results/retention_eval" \
    DATASET_DIR="${DATASET_DIR}" \
    GUIDANCE_MODE="off" \
    TRAJ_SCALE="1.0" \
    MAX_ITEMS="16" \
    DEVICE="cuda:0" \
    bash "${REPO_DIR}/scripts/model-m2diffuser/goal-reach/run_retention_eval.sh" \
        "${ckpt_dir}" \
        "${run_tag}" \
        "${diffuser}" \
        "${pose_cfg}" \
        "${joint_cfg}" \
        2>&1 | tee -a "${EVAL_LOG}"
}

{
    echo "[eval] waiting for ${TRAIN_SESSION} to finish at $(date)"
    wait_for_training_end
    echo "[eval] training finished, starting subset eval at $(date)"

    export PATH=/home/lht/anaconda3/bin:$PATH
    eval "$(/home/lht/anaconda3/condabin/conda shell.bash hook)"
    conda activate m2diffuser
    cd "${REPO_DIR}"

    run_eval "R005_warm_lora" "MK-M2Diffuser-Goal-Reach-CoKin-LoRA" "cokin" "lora_pose_mk" "lora_joint_mk"
    run_eval "R006_warm_adapter" "MK-M2Diffuser-Goal-Reach-CoKin-Adapter" "cokin" "adapter_pose_mk" "adapter_joint_mk"
    run_eval "R007_warm_fadapter" "MK-M2Diffuser-Goal-Reach-CoKin-FAdapter" "cokin" "fadapter_pose_mk" "fadapter_joint_mk"

    echo "[eval] done at $(date)"
} >> "${EVAL_LOG}" 2>&1

