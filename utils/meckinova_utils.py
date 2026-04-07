import copy
import torch
import numpy as np
from trimesh import transform_points
from utils.transform import transform_pointcloud_torch

def transform_trajectory_torch(
    trajs: torch.Tensor,
    trans_mats: torch.Tensor,
    rot_angles: torch.Tensor,
) -> torch.Tensor:
    """ Transform the agent trajectory using transformation matrices and rotation angles.

    Args:
        trajs [torch.Tensor]: Unnormalized agent trajectory, shape [B, L, agent_dof].
        trans_mats [torch.Tensor]: Transformation matrices, shape [B, 4, 4].
        rot_angles [torch.Tensor]: Rotation angles along the z-axis, shape [B, 1].

    Return:
        torch.Tensor: The transformed agent trajectory, shape [B, L, agent_dof].
    """
    trajectories = trajs
    xy_2ds = trajectories[:,:,:2] # [B, L, 2]
    xy_3ds = torch.concatenate(
        (
            xy_2ds, torch.zeros(
                (xy_2ds.shape[0], xy_2ds.shape[1], 1)
            ).to(device=trajectories.device)
        ), 
        dim=-1,
    ) # [B, L, 3]
    xy_3ds = transform_pointcloud_torch(xy_3ds, trans_mats)
    trajectories[:,:,:2] = xy_3ds[:,:,:2]
    trajectories[:,:,2] = trajectories[:,:,2] + rot_angles.unsqueeze(1)
    return trajectories

def transform_configuration_torch(
    cfgs: torch.Tensor,
    trans_mats: torch.Tensor,
    rot_angles: torch.Tensor,
) -> torch.Tensor:
    """ Transform the agent configuration using transformation matrices and rotation angles.

    Args:
        cfgs [torch.Tensor]: Unnormalized agent configurations, shape [B, agent_dof].
        trans_mats [torch.Tensor]: Transformation matrices, shape [B, 4, 4].
        rot_angles [torch.Tensor]: Rotation angles around the z-axis, shape [B, 1].

    Return:
        torch.Tensor: The transformed agent configurations, shape [B, agent_dof].
    """
    B = cfgs.shape[0]
    configurations = cfgs
    xy_2d = configurations[:,:2] # [B, 2]
    xy_3d = torch.concatenate((xy_2d, torch.zeros((B, 1)).to(device=configurations.device)), dim=-1) # [B, 3]
    xy_3d = transform_pointcloud_torch(xy_3d.unsqueeze(1), trans_mats).squeeze(1)
    configurations[:,:2] = xy_3d[:,:2]
    configurations[:,2] = configurations[:,2] + rot_angles
    return configurations

def transform_trajectory_numpy(
    traj: np.ndarray,
    trans_mat: np.ndarray,
    rot_angle: np.ndarray,
) -> np.ndarray:
    """ Transform the agent trajectory using transformation matrices and rotation angles.

    Args:
        trajs [np.ndarray]: Unnormalized agent trajectory, shape [L, agent_dof].
        trans_mats [np.ndarray]: Transformation matrices, shape [4, 4].
        rot_angles [np.ndarray]: Rotation angles along the z-axis, shape [1].

    Return:
        np.ndarray: The transformed agent trajectory, shape [L, agent_dof].
    """
    trajectory = copy.deepcopy(traj)
    xy_2ds = trajectory[:,:2]
    xy_3ds = np.concatenate((xy_2ds, np.zeros((xy_2ds.shape[0], 1))), axis=-1)
    xy_3ds = transform_points(xy_3ds, trans_mat).astype(np.float64)
    trajectory[:,:2] = xy_3ds[:,:2]
    trajectory[:,2] = trajectory[:,2] + rot_angle
    return trajectory

def transform_configuration_numpy(
    cfg: np.ndarray,
    trans_mat: np.ndarray,
    rot_angle: np.ndarray,
) -> np.ndarray:
    """ Transform the agent configuration using transformation matrices and rotation angles.

    Args:
        cfgs [np.ndarray]: Unnormalized agent configurations, shape [agent_dof].
        trans_mats [np.ndarray]: Transformation matrices, shape [4, 4].
        rot_angles [np.ndarray]: Rotation angles around the z-axis, shape [1].

    Return:
        np.ndarray: The transformed agent configurations, shape [agent_dof].
    """
    configuration = copy.deepcopy(cfg)
    xy_2d = configuration[:2]
    xy_3d = np.concatenate((xy_2d, np.zeros(1)), axis=-1)
    xy_3d = np.squeeze(transform_points(np.expand_dims(xy_3d, axis=0), trans_mat).astype(np.float64))
    configuration[:2] = xy_3d[:2]
    configuration[2] = configuration[2] + rot_angle
    return configuration