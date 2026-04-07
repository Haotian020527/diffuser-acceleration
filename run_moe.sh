#!/bin/bash
eval "$(/home/lht/anaconda3/bin/conda shell.bash hook)"
conda activate m2diffuser
cd /home/lht/01Diffuser-acceleration
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/model-m2diffuser/goal-reach/train.sh 2 moe_cokin latest 2>&1 | tee train_m2diffuser_cokin_moe_latest.log
