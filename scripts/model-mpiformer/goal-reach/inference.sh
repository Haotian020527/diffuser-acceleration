CKPT_PATH=$1

python inference_mpiformer.py hydra/job_logging=none hydra/hydra_logging=none \
                exp_dir=${CKPT_PATH} \
                model=mpiformer_mk \
                task=mk_mpiformer_goal_reach \
                task.environment.sim_gui=false \
                task.test.batch_size=1 \
                task.environment.viz_time=10 \