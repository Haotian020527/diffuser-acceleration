#!/bin/bash
set -euo pipefail

BASE_CKPT="/home/lht/01diffuser-acceleration/checkpoints/M1-R004-DenseCoKin/2026-04-09-00-45-57/last.ckpt"
REPO_DIR="/home/lht/01diffuser-acceleration"
LOG_FILE="${REPO_DIR}/experiment_logs/m2/r005_r007_warm.log"
SESSION_NAME="r005_r007_warm"
DATA_DIR="/home/lht/03diffuser-generation/dataset/goal-reach-32demo"

mkdir -p "${REPO_DIR}/experiment_logs/m2"

screen -dmS "${SESSION_NAME}" bash -lc "\
export PATH=/home/lht/anaconda3/bin:\$PATH && \
eval \"\$(/home/lht/anaconda3/condabin/conda shell.bash hook)\" && \
conda activate m2diffuser && \
cd ${REPO_DIR} && \
echo '[launch] base_ckpt=${BASE_CKPT}' > ${LOG_FILE} 2>&1 && \
echo '[launch] data_dir=${DATA_DIR}' >> ${LOG_FILE} 2>&1 && \
for mode in lora_cokin adapter_cokin fadapter_cokin; do \
  echo \"==== Launching \${mode} with warm-start and 32demo split ==== \" >> ${LOG_FILE} 2>&1; \
  INIT_FROM_CKPT=${BASE_CKPT} CUDA_VISIBLE_DEVICES=1 bash ./scripts/model-m2diffuser/goal-reach/train.sh \
    1 \${mode} none no_logging=true task.datamodule.data_dir=${DATA_DIR} \
    >> ${LOG_FILE} 2>&1; \
done"

screen -ls
