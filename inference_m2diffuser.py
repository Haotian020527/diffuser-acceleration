import os
import time
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf
from env.agent.mec_kinova import MecKinova
from env.base import create_enviroment
from utils.meckinova_utils import transform_trajectory_torch
from utils.misc import compute_model_dim
from datamodule.base import create_datamodule
from models.base import create_model

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def run_inference(config: DictConfig) -> None:
    ## compute modeling dimension according to task
    config.model.d_x = compute_model_dim(config.task) 
    if os.environ.get('SLURM') is not None:
        config.slurm = True # update slurm config
    
    device = f'cuda:0' if config.gpus is not None else 'cpu'

    ## prepare test dataset for evaluating on planning task
    dm = create_datamodule(cfg=config.task.datamodule, slurm=config.slurm)
    dl = dm.get_test_dataloader()

    ## create model and diffuser, load ckpt, create and load optimizer and planner for diffuser
    ckpt_path = os.path.join(config.exp_dir, "last.ckpt")
    mdl = create_model(config, ckpt_path=ckpt_path, slurm=config.slurm, **{"device": device})
    mdl.to(device=device)

    ## create meckinova motion policy test environment
    env = create_enviroment(config.task.environment)

    ## inference
    with torch.no_grad():
        mdl.eval()
        for i, data in enumerate(dl):
            # if i > 621:
            #     continue
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            start_time = time.time()
            ## the model outputs have the shape <batch_size, k_sample, denoising_step_T, horizon, dim>
            outputs = mdl.sample(data, k=1) # <B, k, T, L, D>

            ## the agent trajectory is moved from the center of the scene point cloud to the agent initial frame
            traj_norm_a = outputs[:, -1, -1, :, :]
            # traj_unorm_a = MecKinova.unnormalize_joints(torch.clamp(traj_norm_a, min=-1, max=1))
            traj_unorm_a = MecKinova.unnormalize_joints(traj_norm_a)
            traj_unorm_a = transform_trajectory_torch(traj_unorm_a, torch.inverse(data['trans_mat']), -data['rot_angle'])
            traj_unorm_a = traj_unorm_a.squeeze(0).clone().detach().cpu().numpy()
            
            ## evaluate trajectory
            env.evaluate(
                id=i,
                dt=0.08,  # we assume the time step for the trajectory is 0.08
                time=time.time() - start_time,
                data=data, traj=traj_unorm_a, agent_object=MecKinova
            )
            ## visualize trajectory
            env.visualize(data, traj_unorm_a)
        print("Overall Metrics")
        env.print_overall_metrics()


if __name__ == '__main__':
    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_inference()
