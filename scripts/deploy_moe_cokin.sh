#!/bin/bash
# deploy_moe_cokin.sh - Launch MoE CoKin training on GPU server
# Usage: Run this script on the remote server directly

export PATH=/home/lht/anaconda3/bin:$PATH
eval "$(/home/lht/anaconda3/condabin/conda shell.bash hook)"
conda activate m2diffuser

cd /home/lht/01Diffuser-acceleration

echo "=== Starting MoE CoKin training at $(date) ==="
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Use 1 GPU to avoid DDP find_unused_parameters issues with MoE sparse expert selection
bash ./scripts/model-m2diffuser/goal-reach/train.sh 1 moe_cokin 2>&1 | tee /home/lht/01Diffuser-acceleration/train_moe_cokin.log

echo "=== Training finished at $(date) ==="
