EXP_NAME="MK-MPiNets-Goal-Reach"

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
                model=mpinets_mk \
                task=mk_mpinets_goal_reach \