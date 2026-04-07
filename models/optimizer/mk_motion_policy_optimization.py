import torch
import copy
import torch.nn.functional as F
from typing import Dict
from omegaconf import DictConfig
from env.agent.mec_kinova import MecKinova
from env.sampler.mk_sampler import MecKinovaSampler
from models.optimizer.optimizer import Optimizer
from models.base import OPTIMIZER
from utils.meckinova_utils import transform_trajectory_torch
from utils.transform import transform_pointcloud_torch


@OPTIMIZER.register()
class MKMotionPolicyOptimizer(Optimizer):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cpu'
        
        self.scale = cfg.scale
        self.scale_type = cfg.scale_type

        self.collision = cfg.collision
        self.collision_weight = cfg.collision_weight
        self.collision_margin = cfg.collision_margin

        self.joint_limits = cfg.joint_limits
        self.joint_limits_weight = cfg.joint_limits_weight
        self.joint_limits_threshold = cfg.joint_limits_threshold

        self.action_limit = cfg.action_limit
        self.action_limit_weight = cfg.action_limit_weight
        self.action_limit_step = cfg.action_limit_step

        self.smoothness = cfg.smoothness
        self.smoothness_weight = cfg.smoothness_weight

        self.clip_grad_by_value = cfg.clip_grad_by_value

        self.sampler =  MecKinovaSampler(self.device, num_fixed_points=1024, use_cache=True)
    
    def optimize(self, x: torch.Tensor, data: Dict) -> torch.Tensor:
        """ Compute gradient for optimizer constraint.

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
        trans_mats = data['trans_mat'] # [B, 4, 4]
        trans_mats_inv = torch.inverse(trans_mats) # [B, 4, 4]
        rot_angles = data['rot_angle'] # [B]
        rot_angles_inv = -rot_angles
        trajs_unorm = transform_trajectory_torch(trajs_unorm, trans_mats_inv, rot_angles_inv)

        ## load data
        ## NOTE: use copy.deepcopy()
        T_aw = copy.deepcopy(data['T_aw']) # [B, 4, 4]
        sdf_norm_value = copy.deepcopy(data['sdf_norm_value']) # [B, grid_num, grid_num, grid_num]
        scene_mesh_center = copy.deepcopy(data['scene_mesh_center']) # [B, 3]
        scene_mesh_scale = copy.deepcopy(data['scene_mesh_scale']) # [B]
        
        ## compute collision optimization
        if self.collision: 
            collision_loss = 0
            # the last frame of piking and placing motion needs collision
            trajs_unorm_flatten = trajs_unorm[:,:-1,:].reshape(-1, D) # [B×(L-1), D]
            agent_pcs_a = self.sampler.sample(trajs_unorm_flatten) # [B×(L-1), N3, 3]
            agent_pcs_w = transform_pointcloud_torch(
                agent_pcs_a, 
                T_aw.unsqueeze(1).repeat(1, L-1, 1, 1).reshape(-1, 4, 4)
            )
            agent_pcs_w = agent_pcs_a.reshape(B, L-1, -1, 3) # [B, (L-1), N3, 3]
            norm_agent_pcs_w = (
                agent_pcs_w - scene_mesh_center.unsqueeze(1).unsqueeze(1)
            ) * scene_mesh_scale.unsqueeze(1).unsqueeze(1).unsqueeze(1) # [B, (L-1), N3, 3]

            for l in range(L-1):
                norm_agent_pc_w = norm_agent_pcs_w[:,l,:,:].reshape(B, -1, 3) # [B, N3, 3]
                agent_points_num = norm_agent_pc_w.shape[1] # N3
                agent_sdf_batch = F.grid_sample(
                    sdf_norm_value.unsqueeze(1), 
                    norm_agent_pc_w[:,:,[2,1,0]].view(-1, agent_points_num, 1, 1, 3),
                    padding_mode='border', 
                    align_corners=False,
                )

                case1 = (agent_sdf_batch < 0).float()
                result1 = -agent_sdf_batch + 0.5 * self.collision_margin
                case2 = ((agent_sdf_batch >= 0) & (agent_sdf_batch <= self.collision_margin)).float()
                result2 = 0.5 / self.collision_margin * (agent_sdf_batch - self.collision_margin) ** 2
                case3 = (agent_sdf_batch > self.collision_margin).float()
                result3 = torch.zeros_like(agent_sdf_batch)
                collision_loss += (case1 * result1 + case2 * result2 + case3 * result3).mean()
                
            loss += self.collision_weight * collision_loss

        ## compute action limit loss
        if self.action_limit:
            action_limit_loss = 0.0
            actions = trajs_unorm[:,1:,:] - trajs_unorm[:,:-1,:]
            actions_abs = torch.abs(actions)
            actions_limit = torch.as_tensor(MecKinova.ACTION_LIMITS, device=self.device)
            actions_limit = torch.min(torch.abs(actions_limit), dim=-1).values
            action_limit_loss += F.relu(actions_abs - actions_limit).mean()
            loss += self.action_limit_weight * action_limit_loss

        ## compute joint limit loss
        if self.joint_limits:
            joint_limits_loss = 0
            joint_limits = torch.as_tensor(MecKinova.JOINT_LIMITS, device=self.device)
            lower_limit = joint_limits[:,0] * self.joint_limits_threshold
            upper_limit = joint_limits[:,1] * self.joint_limits_threshold
            joint_limits_loss += (F.relu(lower_limit - trajs_unorm).pow(2) + F.relu(trajs_unorm - upper_limit).pow(2)).mean()
            loss += self.joint_limits_weight * joint_limits_loss
        
        ## compute smoothness loss
        if self.smoothness:
            vel = trajs_unorm[:,1:,:] - trajs_unorm[:,:-1,:]
            acc = vel[:,1:,:] - vel[:,:-1,:]
            base_smoothness_loss = (acc[:,:MecKinova.BASE_DOF,:] ** 2).mean()
            arm_smoothness_loss = (acc[:,MecKinova.BASE_DOF:,:] ** 2).mean()
            smoothness_loss = (base_smoothness_loss * 2 + arm_smoothness_loss) / 2
            loss += self.smoothness_weight * smoothness_loss

        return (-1.0) * loss

    def gradient(self, x: torch.Tensor, data: Dict, variance: torch.Tensor) -> torch.Tensor:
        """ Compute gradient for optimizer constraint.

        Args:
            x [torch.Tensor]: the denosied signal at current step
            data [Dict]: data dict that provides original data
            variance [torch.Tensor]: variance at current step
        
        Return:
            torch.Tensor. Commputed gradient.
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            obj = self.optimize(x_in, data)
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
