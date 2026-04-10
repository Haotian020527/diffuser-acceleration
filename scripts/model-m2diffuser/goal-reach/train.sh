EXP_NAME_BASE="MK-M2Diffuser-Goal-Reach"

NUM_GPUS=$1
MODE=${2:-cokin}  # cokin | ddpm | moe_cokin | lora_cokin | adapter_cokin | fadapter_cokin
RESUME_FROM_CKPT=${3:-none}  # none | latest | /abs/path/to/*.ckpt | /abs/path/to/run_dir
EXTRA_ARGS=("${@:4}")
INIT_FROM_CKPT=${INIT_FROM_CKPT:-none}  # /abs/path/to/*.ckpt | /abs/path/to/run_dir

if [ -z "$NUM_GPUS" ]; then
    echo "Usage: ./train.sh <num_gpus> [cokin|ddpm|moe_cokin|lora_cokin|adapter_cokin|fadapter_cokin] [none|latest|ckpt_path|run_dir] [extra hydra overrides ...]"
    echo "Optional warm-start: set INIT_FROM_CKPT=/abs/path/to/base.ckpt (or checkpoint dir)."
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

INIT_ARGS=()
if [ "${INIT_FROM_CKPT}" != "none" ]; then
    INIT_ARGS=("init_from_checkpoint=${INIT_FROM_CKPT}")
    echo "Warm-start enabled: ${INIT_FROM_CKPT}"
fi

if [[ "${MODE}" =~ ^(lora_cokin|adapter_cokin|fadapter_cokin)$ ]] && [ "${INIT_FROM_CKPT}" = "none" ]; then
    echo "Warning: ${MODE} will train with a randomly initialized frozen backbone unless INIT_FROM_CKPT is set."
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
                    "${INIT_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
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
                    "${INIT_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
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
                    "${INIT_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
                    "${RESUME_ARGS[@]}"
elif [ "${MODE}" = "lora_cokin" ]; then
    python train.py hydra/job_logging=none hydra/hydra_logging=none \
                    exp_name=${EXP_NAME_BASE}-CoKin-LoRA \
                    gpus="${GPUS}" \
                    diffuser=cokin \
                    diffuser.loss_type=l2 \
                    diffuser.timesteps=50 \
                    model=m2diffuser_mk \
                    +model@pose_model=lora_pose_mk \
                    +model@joint_model=lora_joint_mk \
                    model.use_position_embedding=true \
                    task=mk_m2diffuser_goal_reach \
                    task.train.batch_size=32 \
                    task.train.num_epochs=2000 \
                    "${INIT_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
                    "${RESUME_ARGS[@]}"
elif [ "${MODE}" = "adapter_cokin" ]; then
    python train.py hydra/job_logging=none hydra/hydra_logging=none \
                    exp_name=${EXP_NAME_BASE}-CoKin-Adapter \
                    gpus="${GPUS}" \
                    diffuser=cokin \
                    diffuser.loss_type=l2 \
                    diffuser.timesteps=50 \
                    model=m2diffuser_mk \
                    +model@pose_model=adapter_pose_mk \
                    +model@joint_model=adapter_joint_mk \
                    model.use_position_embedding=true \
                    task=mk_m2diffuser_goal_reach \
                    task.train.batch_size=32 \
                    task.train.num_epochs=2000 \
                    "${INIT_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
                    "${RESUME_ARGS[@]}"
elif [ "${MODE}" = "fadapter_cokin" ]; then
    python train.py hydra/job_logging=none hydra/hydra_logging=none \
                    exp_name=${EXP_NAME_BASE}-CoKin-FAdapter \
                    gpus="${GPUS}" \
                    diffuser=cokin \
                    diffuser.loss_type=l2 \
                    diffuser.timesteps=50 \
                    +diffuser.low_band_anchor_weight=0.5 \
                    +diffuser.band_leak_weight=0.05 \
                    +diffuser.anchor_pose_weight=1.0 \
                    +diffuser.anchor_joint_weight=0.5 \
                    model=m2diffuser_mk \
                    +model@pose_model=fadapter_pose_mk \
                    +model@joint_model=fadapter_joint_mk \
                    model.use_position_embedding=true \
                    task=mk_m2diffuser_goal_reach \
                    task.train.batch_size=32 \
                    task.train.num_epochs=2000 \
                    "${INIT_ARGS[@]}" \
                    "${EXTRA_ARGS[@]}" \
                    "${RESUME_ARGS[@]}"
else
    echo "Unsupported mode: ${MODE}. Use cokin, ddpm, moe_cokin, lora_cokin, adapter_cokin, or fadapter_cokin."
    exit 1
fi
