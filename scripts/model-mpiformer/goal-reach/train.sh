EXP_NAME="MK-MPiFormer-Goal-Reach"

NUM_GPUS=$1

if [ -z "$NUM_GPUS" ]; then
    echo "Usage: ./train.sh <num_gpus>"
    exit 1
fi

GPUS="["
for ((i=0; i<NUM_GPUS; i++)); do
    if [ $i -gt 0 ]; then
        GPUS+=","
    fi
    GPUS+="$i"
done
GPUS+="]"

echo "Launching training on GPUs: ${GPUS}"

python train.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_name=${EXP_NAME} \
                gpus="${GPUS}" \
                model=mpiformer_mk \
                model.loss.collision_loss_weight=5 \
                model.loss.point_match_loss_weight=1 \
                task=mk_mpiformer_goal_reach \