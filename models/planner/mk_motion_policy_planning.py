import copy
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import kaolin as kl
from argparse import Namespace
from typing import Dict
from omegaconf import DictConfig
from env.agent.mec_kinova import MecKinova
from env.sampler.mk_sampler import MecKinovaSampler
from models.base import PLANNER
from models.planner.planner import Planner
from third_party.grasp_diffusion.se3dif.models.loader import load_model
from utils.meckinova_utils import transform_trajectory_torch
from utils.transform import SE3, transform_pointcloud_torch


@PLANNER.register()
class MKMotionPolicyPlanner(Planner):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type

        self.grasp_energy = cfg.grasp_energy
        self.grasp_energy_weight = cfg.grasp_energy_weight
        self.grasp_energy_type = cfg.grasp_energy_type
        self.grasp_energy_model = cfg.grasp_energy_model
        self.grasp_gripper_type = cfg.grasp_gripper_type
        self.grasp_energy_lowerlimit = cfg.grasp_energy_lowerlimit
        self.grasp_energy_len = None

        self.place_energy = cfg.place_energy
        self.place_extra_height = cfg.place_extra_height
        self.place_energy_weight = cfg.place_energy_weight
        self.place_energy_method = cfg.place_energy_method
        self.place_energy_type = cfg.place_energy_type

        self.goal_reach_energy = cfg.goal_reach_energy
        self.goal_reach_energy_weight = cfg.goal_reach_energy_weight
        self.goal_reach_energy_method = cfg.goal_reach_energy_method
        self.goal_reach_energy_type = cfg.goal_reach_energy_type

        self.clip_grad_by_value = cfg.clip_grad_by_value

        assert at_most_one_true(self.grasp_energy, self.place_energy, self.goal_reach_energy), \
                "Grasping, Placement or Goal-reaching, Only Choose One Task."

        self.sampler = MecKinovaSampler(self.device, num_fixed_points=32768, use_cache=True)

    def objective(self, x: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Compute gradient for planner guidance.

        Args:
            x [torch.Tensor]: the denosied signal at current step, which is detached and is required grad.
            data [Dict]: data dict that provides original data.

        Return:
            torch.Tensor. The optimizer objective value of current step.
        """
        loss = 0

        x = x + 0.0 # only non-leaf nodes can calculate gradients
        _, O, _ = data['start'].shape 
        x[:, 0:O, :] = data['start'].clone() # copy start observation to x after unnormalize

        B, L, D = x.shape # batch_size, trajectory_length, agent_dof

        ## important!!!
        ## normalize x and convert it to representation in agent initial frame
        trajs_norm = x # [B, L, D]
        trajs_unorm = MecKinova.unnormalize_joints(trajs_norm) # [B, L, D]
        trajs_unorm = transform_trajectory_torch(
            trajs_unorm, torch.inverse(data['trans_mat']), -data['rot_angle']
        )

        ## compute grasp pose energy loss
        if self.grasp_energy:
            grasp_energy_loss = 0
            ## load data
            ## NOTE: use copy.deepcopy()
            object_pcs_a = copy.deepcopy(data['object_pc_a']) # [B, N2, 3]
            T_aw = copy.deepcopy(data['T_aw']) # [B, 4, 4]
            T_ow = copy.deepcopy(data['T_ow']) # [B, 4, 4]
            T_oa = copy.deepcopy(data['T_oa']) # [B, 4, 4]

            if self.grasp_energy_type == "last_frame":
                self.grasp_energy_len = 1
            elif self.grasp_energy_type == "all_frame" or self.grasp_energy_type == "all_frame_exp":
                self.grasp_energy_len = L
            else:
                raise Exception('Unsupported energy type')
            
            T_ao = torch.inverse(T_oa)
            T_wo = torch.inverse(T_ow)
            object_pcs_o = transform_pointcloud_torch(object_pcs_a, T_ao)

            # ## visualize point clouds
            # import open3d as o3d
            # points = object_pc_o[0].clone().detach().cpu().numpy()
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)
            # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
            # o3d.visualization.draw_geometries([point_cloud, frame])

            # normalize object point clouds
            object_pcs_o *= 8.
            object_pcs_o_mean = torch.mean(object_pcs_o, dim=1).unsqueeze(1)
            object_pcs_o -= object_pcs_o_mean

            for b in range(B):
                object_pc_o = object_pcs_o[b,...]
                # load grasp-energy model 
                args = Namespace(device=self.device, model=self.grasp_energy_model, batch=self.grasp_energy_len)
                model_args = {'device': self.device, 'pretrained_model': args.model}
                grasp_energy_model = load_model(model_args)
                context = object_pc_o[None,...]
                grasp_energy_model.set_latent(context, batch=args.batch)
                grasp_energy_model.eval()

                # compute grasping pose energy
                if self.grasp_energy_type == "last_frame":
                    H_a = self.sampler.end_effector_pose(trajs_unorm[b,...][-1]).squeeze(0)
                    H_w = torch.matmul(T_aw[b,...], H_a)
                    H_o = torch.matmul(T_wo[b,...], H_w)
                    # normalize SE(3) transformation matrix
                    H_o[:3, -1] *= 8
                    H_o[:3, -1] = H_o[:3, -1] - object_pcs_o_mean[b,...]
                    # grasp energy
                    _, grasp_energy = compute_grasp_energy(
                        grasp_energy_model, H_o[None,...], energy_lowerlimit=self.grasp_energy_lowerlimit
                    )
                    grasp_energy_loss += grasp_energy.mean()
                elif self.grasp_energy_type == "all_frame" or self.grasp_energy_type == "all_frame_exp":
                    H_a = self.sampler.end_effector_pose(trajs_unorm[b,...])
                    H_w = torch.matmul(T_aw[b,...], H_a)
                    H_o = torch.matmul(T_wo[b,...], H_w)
                    # normalize SE(3) transformation matrix
                    H_o[..., :3, -1] *= 8
                    H_o[..., :3, -1] = H_o[..., :3, -1] - object_pcs_o_mean[b,...]
                    # grasp energy
                    _, grasp_energy = compute_grasp_energy(
                        grasp_energy_model, H_o, energy_lowerlimit=self.grasp_energy_lowerlimit
                    )
                    if self.grasp_energy_type == "all_frame":
                        grasp_energy_loss += grasp_energy.mean()
                    else:
                        grasp_energy_loss += (-1.0) * torch.exp(1 / grasp_energy).mean()
                else:
                    raise Exception('Unsupported grasp energy type')
            loss += self.grasp_energy_weight * grasp_energy_loss
        
        ## compute object place energy loss
        if self.place_energy:
            plcae_energy_loss = 0
            # 'scene_placement_pc_a' keeps stationary in the world or agent frame
            scene_placement_pcs_a = copy.deepcopy(data['scene_placement_pc_a'])
            scene_placement_pcs_a[...,-1] += self.place_extra_height
            # 'object_placement_pc_a' follows the agent changes in real time
            object_placement_pcs_o = copy.deepcopy(data['object_placement_pc_o'])
            Ts_eeo = copy.deepcopy(data['grasping_pose'])

            for b in range(B):
                if self.place_energy_type == "last_frame":
                    T_eea = self.sampler.end_effector_pose(trajs_unorm[b,...][-1]) # [1, 4, 4]
                    T_oa = torch.matmul(T_eea, torch.inverse(Ts_eeo[b,...])) # [1, 4, 4]
                    object_placement_pc_a = transform_pointcloud_torch(object_placement_pcs_o[b,...].unsqueeze(0), T_oa)
                    scene_placement_pc_a = scene_placement_pcs_a[b,...].unsqueeze(0)
                    if self.place_energy_method == 'sided_distance':
                        distance, _ = kl.metrics.pointcloud.sided_distance(object_placement_pc_a, scene_placement_pc_a)
                    else:
                        distance = kl.metrics.pointcloud.chamfer_distance(object_placement_pc_a, scene_placement_pc_a)
                    plcae_energy_loss += distance.mean()
                elif self.place_energy_type == "all_frame" or self.place_energy_type == "all_frame_exp":
                    T_eea = self.sampler.end_effector_pose(trajs_unorm[b,...]) # [L, 4, 4]
                    T_oa = torch.matmul(T_eea, torch.inverse(Ts_eeo[b,...]).repeat(L, 1, 1))
                    object_placement_pc_a = transform_pointcloud_torch(object_placement_pcs_o[b,...].repeat(L, 1, 1), T_oa)
                    scene_placement_pc_a = scene_placement_pcs_a[b,...].repeat(L, 1, 1)
                    if self.place_energy_method == 'sided_distance':
                        distance, _ = kl.metrics.pointcloud.sided_distance(object_placement_pc_a, scene_placement_pc_a)
                    else:
                        distance = kl.metrics.pointcloud.chamfer_distance(object_placement_pc_a, scene_placement_pc_a)
                    if self.place_energy_type == "all_frame":
                        plcae_energy_loss += distance.mean()
                    else:
                        plcae_energy_loss += (-1.0) * torch.exp(1 / distance).mean()
                else:
                    raise Exception('Unsupported place energy type')
            loss += self.place_energy_weight * plcae_energy_loss
        
        ## compute goal-reaching energy losss
        if self.goal_reach_energy:
            goal_reach_energy_loss = 0
            # 'target_pcs_a' keeps stationary in the world or agent frame
            target_pcs_a = copy.deepcopy(data['target_pc_a']) # [B, N, 3]
            # the number of target's points
            num_target_pc = target_pcs_a.shape[1]

            for b in range(B):
                if self.goal_reach_energy_type == "last_frame":
                    eef_pc_a = self.sampler.sample_end_effector(trajs_unorm[b,...][-1], num_target_pc, True) # [1, N, 3]
                    target_pc_a = target_pcs_a[b,...].unsqueeze(0) # [1, N, 3]
                    if self.goal_reach_energy_method == 'points_distance':
                        distance = F.mse_loss(eef_pc_a, target_pc_a, reduction='none') + \
                                    F.l1_loss(eef_pc_a, target_pc_a, reduction='none')
                    elif self.goal_reach_energy_method == 'sided_distance':
                        distance, _ = kl.metrics.pointcloud.sided_distance(eef_pc_a, target_pc_a)
                    else:
                        distance = kl.metrics.pointcloud.chamfer_distance(eef_pc_a, target_pc_a)
                    goal_reach_energy_loss += distance.mean()
                elif self.goal_reach_energy_type == "all_frame" or self.goal_reach_energy_type == "all_frame_exp":
                    eef_pc_a = self.sampler.sample_end_effector(trajs_unorm[b,...], num_target_pc, True) # [L, N, 3]
                    target_pc_a = target_pcs_a[b,...].repeat(L, 1, 1) # [L, N, 3]
                    if self.goal_reach_energy_method == 'points_distance':
                        distance = F.mse_loss(eef_pc_a, target_pc_a, reduction='none') + \
                                    F.l1_loss(eef_pc_a, target_pc_a, reduction='none')
                    elif self.goal_reach_energy_method == 'sided_distance':
                        distance, _ = kl.metrics.pointcloud.sided_distance(eef_pc_a, target_pc_a)
                    else:
                        distance = kl.metrics.pointcloud.chamfer_distance(eef_pc_a, target_pc_a)

                    if self.goal_reach_energy_type == "all_frame":
                        goal_reach_energy_loss += distance.mean()
                    else:
                        goal_reach_energy_loss += (-1.0) * torch.exp(1 / distance).mean()
                else:
                    raise Exception('Unsupported goal-reach energy type')
            loss += self.goal_reach_energy_weight * goal_reach_energy_loss

        return (-1.0) * loss
    
    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        """ Compute gradient for planner guidance.

        Args:
            x [torch.Tensor]: the denosied signal at current step.
            data [Dict]: data dict that provides original data.

        Return:
            torch.Tensor. Commputed gradient.
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.objective(x_in, data)
            grad = torch.autograd.grad(obj, x_in)[0]
            ## clip gradient by value
            grad = torch.clip(grad, **self.clip_grad_by_value)
            ## TODO clip gradient by norm

            if self.scale_type == 'normal':
                grad = self.scale * grad * variance
            elif self.scale_type == 'div_var':
                grad = self.scale * grad
            else:
                raise Exception('Unsupported scale type!')

            return grad

def at_most_one_true(x, y, z):
    return (x and not y and not z) or (not x and y and not z) or \
            (not x and not y and z) or (not x and not y and not z)

def marginal_prob_std(t, sigma=0.5):
    return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

def compute_grasp_energy(model: torch.nn.Module, H: torch.Tensor, t: int=0, T: int=50, energy_lowerlimit: int=-150) -> torch.Tensor:
    """ This function computes the grasp energy and its associated loss for a given SE(3) transformation matrix using a provided model. 
    It calculates the phase of the denoising process, computes the energy coefficient, and evaluates the energy and loss based on the model's output.

    Args:
        model [torch.nn.Module]: The neural network model used to predict energy values.
        H [torch.Tensor]: SE(3) transformation matrix representing the pose(s).
        t [int]: Current denoising step.
        T [int]: Total number of denoising steps.
        energy_lowerlimit [int]: The lower limit for the energy value.

    Return:
        Tuple[torch.Tensor, torch.Tensor]: 
            - The predicted energy tensor from the model.
            - The computed energy loss tensor.
    """
    # phase
    eps = 1e-3
    phase = t / T + eps
    sigma_T = marginal_prob_std(eps)

    # energy and coefficient
    alpha = 1e-3
    sigma_i = marginal_prob_std(phase)
    ratio = sigma_i ** 2 / sigma_T ** 2
    c_lr = alpha * ratio

    if t == 0:
        c_lr = 0.003

    t_in = phase * torch.ones_like(H[:,0,0])
    e = model(H, t_in)
    e_loss = (e - energy_lowerlimit) ** 2 * c_lr

    return e, e_loss