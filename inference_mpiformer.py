import copy
import os
import time
import hydra
import torch
import random
import numpy as np
import torch.nn as nn
import kaolin as kl
from argparse import Namespace
from typing import Dict
from omegaconf import DictConfig, OmegaConf
from env.agent.mec_kinova import MecKinova
from env.base import create_enviroment
from env.sampler.mk_sampler import MecKinovaSampler
from models.planner.mk_motion_policy_planning import compute_grasp_energy
from third_party.grasp_diffusion.se3dif.models.loader import load_model
from utils.meckinova_utils import transform_configuration_torch
from utils.misc import compute_model_dim
from datamodule.base import create_datamodule
from models.base import create_model
from utils.transform import transform_pointcloud_torch

OmegaConf.register_new_resolver("eval", eval, replace=True)

def rollout_mpiformer(
    mdl: nn.Module,
    data: Dict[str, torch.Tensor],
    mk_sampler: MecKinovaSampler,
    task_cfg: dict,
    num_pc_item: Dict[str, int],
) -> np.ndarray:
    """ This function performs a rollout of the MPiFormer policy model for a given task, 
    generating a trajectory of robot configurations based on the initial observation and 
    task configuration. It supports different task types such as 'pick', 'place', and 
    'goal-reach', and selects the optimal trajectory point according to task-specific 
    energy metrics (e.g., grasping energy, placement energy, or goal-reaching energy).

    Args:
        mdl [nn.Module]: The policy model used to predict the next configuration in the trajectory.
        data [Dict[str, torch.Tensor]]: Input data containing initial configurations, observations, transformation matrices, and point clouds 
            required for the rollout.
        mk_sampler [MecKinovaSampler]: A sampler for generating robot surface points and end-effector poses.
        task_cfg [dict]: Task configuration dictionary specifying task type and rollout parameters.
        num_pc_item [Dict[str, int]]: Dictionary specifying the number of points for agent, scene, object, placement area, and target.

    Return:
        np.ndarray: The generated trajectory of robot configurations for each batch.
        np.ndarray: The indices of the trajectory steps with the minimum energy (grasping, placement, or goal-reaching) for each batch.
    """
    # [batch_size, context_length, agent_dof]
    B, C = data['configuration_sq'].shape[0], data['configuration_sq'].shape[1]
    
    # initial configuration and observation
    q0_norm = data['configuration_sq'][:,0,:] # [B, D]
    obser0 = data['xyz_sq'][:,0,:,:] # [B, N, 4]
    device = q0_norm.device

    # load task information
    task_type = task_cfg.type
    L = task_cfg.max_predicted_length
    a_num = num_pc_item['num_agent_points']
    s_num = num_pc_item['num_scene_points']
    o_num = num_pc_item['num_object_points']
    p_num = num_pc_item['num_placement_area_points']
    t_num = num_pc_item['num_target_points']

    # transformation matrices
    T_aw = copy.deepcopy(data['T_aw']) # [B, 4, 4]
    if task_type == 'pick':
        T_ow = copy.deepcopy(data['T_ow']) # [B, 4, 4]
        T_wo = torch.inverse(T_ow) # [B, 4, 4]
        T_oa = copy.deepcopy(data['T_oa']) # [B, 4, 4]
        T_ao = torch.inverse(T_oa) # [B, 4, 4]
    elif task_type == 'place':
        T_ow_final = copy.deepcopy(data['T_ow_final'])
        T_wo_final = torch.inverse(T_ow_final)
        T_oa_init = copy.deepcopy(data['T_oa_init'])
        T_ao_init = torch.inverse(T_oa_init)
        T_eeo = copy.deepcopy(data['grasping_pose'])

    # load object point cloud
    if task_type == 'pick':
        object_pcs_a = copy.deepcopy(data['object_pc_a']) # [B, N2, 3]
    elif task_type == 'place':
        # initial object point cloud in agent frame
        object_pcs_a = copy.deepcopy(data['object_pc_a']) # [B, N2, 3]
        # object point cloud in self frame
        object_pcs_o = transform_pointcloud_torch(object_pcs_a, T_ao_init)
        object_placement_pcs_o = copy.deepcopy(data['object_placement_pc_o'])
        scene_placement_pcs_a = copy.deepcopy(data['scene_placement_pc_a'])
    else:
        target_pcs_a = copy.deepcopy((data['target_pc_a']))

    # transformation in data normalization
    trans_mats = data['trans_mat']
    trans_mats_inv = torch.inverse(trans_mats)
    rot_angles = data['rot_angle']
    rot_angles_inv = -rot_angles

    # unnormalize and transform q0
    q0_unnorm = MecKinova.unnormalize_joints(q0_norm) # [B, D]
    q0_unnorm = transform_configuration_torch(q0_unnorm, trans_mats_inv, rot_angles_inv) # [B, D]

    trajs = [q0_unnorm] # store complete trajectory

    # create model input
    if task_type == 'pick' or task_type == 'goal-reach':
        obs = torch.cat(
            (
                torch.zeros(a_num, 4), # The agent point cloud is labeled 0
                torch.ones(s_num, 4), # The scene point cloud is labeled 1
                2 * torch.ones(o_num, 4), # The object point cloud is labeled 2
            ), dim=0
        ).float().to(device=device)
    elif task_type == 'place':
        obs = torch.cat(
            (
                torch.zeros(a_num, 4), # The agent point cloud is labeled 0
                torch.ones(s_num, 4), # The scene point cloud is labeled 1
                2 * torch.ones(o_num, 4), # The object point cloud is labeled 2
                3 * torch.ones(p_num, 4), # The scene placement point cloud is labeled 3
            ), dim=0
        ).float().to(device=device)
    obs_sq = torch.stack([obs for _ in range(C)], dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # [B, C, N, 4]
    cfg_sq = torch.zeros(B, C, MecKinova.DOF).float().to(device=device) # [B, C, D]
    tim_sq = torch.zeros(B, C).long().to(device=device) # [B, C]
    atm_sq = torch.zeros(B, C).long().to(device=device) # [B, C]

    # updata obser_sq and cfg_seq using initial configuration and observation
    obs_sq = torch.cat([obs_sq, obser0.unsqueeze(1)], dim=1) # [B, C + 1, N, 4]
    cfg_sq = torch.cat([cfg_sq, q0_norm.unsqueeze(1)], dim=1) # [B, C + 1, D]

    for h in range(L - 1):
        # update timesteps and attention mask
        tim_sq = torch.cat([tim_sq, torch.ones(B, 1).long().to(device=device) * h], dim=1)
        atm_sq = torch.cat([atm_sq, torch.ones(B, 1).long().to(device=device) * 1], dim=1)
        # predict next sequence
        q_sq_hat_norm = mdl(
            obs_sq[:,-C:,...], cfg_sq[:,-C:,...], tim_sq[:,-C:,...], atm_sq[:,-C:,...]
        )
        # predicted next configuration
        # q_hat_norm = torch.clamp(q_sq_hat_norm[:,-1,:], min=-1, max=1) # [B, D]
        q_hat_norm = q_sq_hat_norm[:,-1,:] # [B, D]
        q_hat_unnorm = MecKinova.unnormalize_joints(q_hat_norm) # in local scene center frame
        q_hat_unnorm = transform_configuration_torch(q_hat_unnorm, trans_mats_inv, rot_angles_inv) # in agent initial frame
        trajs.append(q_hat_unnorm)

        # updata obser_sq and cfg_seq using predicted configuration
        agent_pcs_a = mk_sampler.sample(q_hat_unnorm).type_as(obs_sq)
        agent_pcs_s = transform_pointcloud_torch(agent_pcs_a, trans_mats)
        obs_sq = torch.cat([obs_sq, obs_sq[:,-1:,...]], dim=1)
        obs_sq[:,-1,:a_num,:-1] = agent_pcs_s
        
        # for placement task, we should update object point cloud
        if task_type == 'place':
            T_eea = mk_sampler.end_effector_pose(q_hat_unnorm) # [B, 4, 4]
            T_oa = torch.matmul(T_eea, torch.inverse(T_eeo)) # [B, 4, 4]
            object_pcs_a = transform_pointcloud_torch(copy.deepcopy(object_pcs_o), T_oa)
            object_pcs_s = transform_pointcloud_torch(object_pcs_a, trans_mats)
            obs_sq[:,-1,a_num + s_num:-p_num,:-1] = object_pcs_s

        cfg_sq = torch.cat([cfg_sq, q_hat_norm.unsqueeze(1)], dim=1)
    
    trajs = np.asarray([t.detach().cpu().numpy() for t in trajs])
    trajs = torch.as_tensor(trajs).to(device=device) # [L, B, D]
    trajs = trajs.transpose(0, 1) # [B, L, D]

    ## for 'pick' task, we search locations with the least grasping energy in each trajectory
    if task_type == 'pick':
        # load grasp-energy model 
        args = Namespace(device=device, model='grasp_dif_multi', batch=L)
        model_args = {'device': device, 'pretrained_model': args.model}
        grasp_energy_model = load_model(model_args)
        object_pcs_o = transform_pointcloud_torch(object_pcs_a, T_ao)
        # normalize object point clouds
        object_pcs_o *= 8.
        object_pcs_o_mean = torch.mean(object_pcs_o, dim=1).unsqueeze(1)
        object_pcs_o -= object_pcs_o_mean # [B, N, 3]
        # calculate the grasping energy for each trajectory
        min_grasp_enery_index = torch.tensor([]).to(device=device)
        for b in range(B):
            object_pc_o = object_pcs_o[b,...]
            context = object_pc_o[None,...]
            grasp_energy_model.set_latent(context, batch=args.batch)
            H_a = mk_sampler.end_effector_pose(trajs[b,...])
            H_w = torch.matmul(T_aw[b,...], H_a)
            H_o = torch.matmul(T_wo[b,...], H_w)
            # normalize SE(3) transformation matrix
            H_o[..., :3, -1] *= 8
            H_o[..., :3, -1] = H_o[..., :3, -1] - object_pcs_o_mean[b,...]
            # grasp energy
            grasp_energy, _ = compute_grasp_energy(grasp_energy_model, H_o)
            _, min_index = torch.min(grasp_energy, dim=0)
            min_index = torch.as_tensor([L], device=device)
            min_grasp_enery_index = torch.cat((min_grasp_enery_index, min_index), dim=0)
        return trajs.detach().cpu().numpy(), min_grasp_enery_index.detach().cpu().numpy().astype(np.int16)
    ## for 'place' task, we search locations with the least placement energy in each trajectory
    elif task_type == 'place':
        # calculate the grasping energy for each trajectory
        min_place_enery_index = torch.tensor([]).to(device=device)
        for b in range(B):
            T_eea = mk_sampler.end_effector_pose(trajs[b,...]) # [L, 4, 4]
            T_oa = torch.matmul(T_eea, torch.inverse(T_eeo[b,...]).repeat(L, 1, 1))
            object_placement_pc_a = transform_pointcloud_torch(
                object_placement_pcs_o[b,...].repeat(L, 1, 1), torch.inverse(T_oa)
            )
            scene_placement_pc_a = scene_placement_pcs_a[b,...].repeat(L, 1, 1)
            place_energy, _ = kl.metrics.pointcloud.sided_distance(
                object_placement_pc_a, scene_placement_pc_a
            )
            _, min_index = torch.min(place_energy, dim=0)
            min_index = torch.as_tensor([L], device=device)
            min_place_enery_index = torch.cat((min_place_enery_index, min_index), dim=0)
        return trajs.detach().cpu().numpy(), min_place_enery_index.detach().cpu().numpy().astype(np.int16)
    ## for 'goal-reach' task, we search locations with the least goal-reaching energy in each trajectory
    else:
        # calculate the grasping energy for each trajectory
        min_goal_reach_enery_index = torch.tensor([]).to(device=device)
        for b in range(B):
            eef_pc_a = mk_sampler.sample_end_effector(trajs[b,...], t_num, True) # [L, N, 3]
            target_pc_a = target_pcs_a[b,...].repeat(L, 1, 1) # [L, N, 3]
            # goal_reach_energy = point_clouds_match_loss(eef_pc_a, target_pc_a, 'none').sum(dim=-1).sum(dim=-1).unsqueeze(1) # [L, 1]
            # _, min_index = torch.min(goal_reach_energy, dim=0)
            goal_reach_energy = kl.metrics.pointcloud.chamfer_distance(eef_pc_a, target_pc_a).unsqueeze(1)
            _, min_index = torch.min(goal_reach_energy, dim=0)
            min_goal_reach_enery_index = torch.cat((min_goal_reach_enery_index, min_index), dim=0)
        return trajs.detach().cpu().numpy(), min_goal_reach_enery_index.detach().cpu().numpy().astype(np.int16)


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

    ## number of point cloud
    num_pc_item = {
        "num_scene_points": config.task.datamodule.num_scene_points,
        "num_agent_points": config.task.datamodule.num_agent_points,
        "num_object_points": config.task.datamodule.num_object_points,
        "num_placement_area_points": config.task.datamodule.num_placement_area_points,
        "num_target_points": config.task.datamodule.num_target_points
    }

    ## inference
    with torch.no_grad():
        mdl.eval()
        for i, data in enumerate(dl):
            for key in data:
                if torch.is_tensor(data[key]):
                    data[key] = data[key].to(device)

            start_time = time.time()
            ## create meckinova sampler
            mk_sampler = MecKinovaSampler(
                device=device, 
                num_fixed_points=num_pc_item['num_agent_points'], 
                use_cache=True,
            )
            ## rolls out the policy until the success criteria are met.
            trajs_unorm_a, termination_index = rollout_mpiformer(
                mdl, data, mk_sampler, config.task, num_pc_item
            )
            ## evaluation mode supports only 1 batch_size
            traj_unorm_a = trajs_unorm_a[-1][:termination_index[-1] + 1]
            ## evaluate trajectory
            env.evaluate(
                id=i,
                dt=0.08,  # we assume the time step for the trajectory is 0.08
                time=time.time() - start_time,
                data=data, traj=traj_unorm_a, agent_object=MecKinova
            )
            env.visualize(data, traj_unorm_a)
        print("Overall Metrics")
        env.print_overall_metrics()


if __name__ == '__main__':
    ## set random seed
    seed = 0
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    run_inference()