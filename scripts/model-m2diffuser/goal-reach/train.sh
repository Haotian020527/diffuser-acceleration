EXP_NAME_BASE="MK-M2Diffuser-Goal-Reach"

NUM_GPUS=$1
MODE=${2:-cokin}  # cokin | ddpm | moe_cokin
RESUME_FROM_CKPT=${3:-none}  # none | latest | /abs/path/to/*.ckpt | /abs/path/to/run_dir

if [ -z "$NUM_GPUS" ]; then
    echo "Usage: ./train.sh <num_gpus> [cokin|ddpm|moe_cokin] [none|latest|ckpt_path|run_dir]"
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

echo "Launching ${MODE} training on GPUs: ${GPUS}"

RESUME_ARGS=()
if [ "${RESUME_FROM_CKPT}" != "none" ]; then
    RESUME_ARGS=("resume_from_checkpoint=${RESUME_FROM_CKPT}")
    echo "Resume enabled: ${RESUME_FROM_CKPT}"
fi

if [ "${MODE}" = "ddpm" ]; then
    python train.py hydra/job_logging=none hydra/hydra_logging=none \
                    exp_name=${EXP_NAME_BASE} \
                    gpus="${GPUS}" \
                    diffuser=ddpm \
                    diffuser.loss_type=l2 \
                    diffuser.timesteps=50 \
                    model=m2diffuser_mk \
                    model.use_position_embedding=true \
                    task=mk_m2diffuser_goal_reach \
                    task.train.num_epochs=2000 \
                    "${RESUME_ARGS[@]}"
elif [ "${MODE}" = "cokin" ]; then
    python train.py hydra/job_logging=none hydra/hydra_logging=none \
                    exp_name=${EXP_NAME_BASE}-CoKin \
                    gpus="${GPUS}" \
                    diffuser=cokin \
                    diffuser.loss_type=l2 \
                    diffuser.timesteps=50 \
                    model=m2diffuser_mk \
                    +model@pose_model=cokin_pose_mk \
                    +model@joint_model=cokin_joint_mk \
                    model.use_position_embedding=true \
                    task=mk_m2diffuser_goal_reach \
                    task.train.num_epochs=2000 \
                    "${RESUME_ARGS[@]}"
elif [ "${MODE}" = "moe_cokin" ]; then
    python train.py hydra/job_logging=none hydra/hydra_logging=none \
                    exp_name=${EXP_NAME_BASE}-CoKin-MoE \
                    gpus="${GPUS}" \
                    diffuser=cokin_moe \
                    diffuser.loss_type=l2 \
                    diffuser.timesteps=50 \
                    model=m2diffuser_mk \
                    +model@pose_model=cokin_moe_pose_mk \
                    +model@joint_model=cokin_moe_joint_mk \
                    model.use_position_embedding=true \
                    task=mk_m2diffuser_goal_reach \
                    task.train.batch_size=32 \
                    task.train.num_epochs=2000 \
                    "${RESUME_ARGS[@]}"
else
    echo "Unsupported mode: ${MODE}. Use cokin, ddpm, or moe_cokin."
    exit 1
fi
