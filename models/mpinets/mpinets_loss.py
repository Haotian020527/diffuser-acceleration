import torch
import torch.nn.functional as F

def point_clouds_match_loss(input_pc: torch.Tensor, target_pc: torch.Tensor, reduction: str="mean") -> torch.Tensor:
    """ This function computes a combined L1 and L2 loss between two point clouds,
    penalizing both large and small deviations. It is typically used to measure
    the similarity between a predicted point cloud and a target (ground truth)
    point cloud in tasks such as 3D reconstruction or generation.

    Args:
        input_pc [torch.Tensor]: The predicted point cloud from the network, with shape [B, N, 3].
        target_pc [torch.Tensor]: The ground truth point cloud for supervision, with shape [B, N, 3].
        reduction [str]: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'. Default is 'mean'.

    Return:
        torch.Tensor: The computed loss value representing the similarity between the input and target point clouds.
    """
    return F.mse_loss(input_pc, target_pc, reduction=reduction) + \
            F.l1_loss(input_pc, target_pc, reduction=reduction)

def sdf_collision_loss(
    agent_pcs: torch.Tensor,
    sdf_norm_values: torch.Tensor,
) -> torch.Tensor:
    """ Calculates the collision loss for a robot represented as a point cloud by evaluating the signed 
    distance function (SDF) values at the agent's points. If any agent points penetrate obstacles (i.e., 
    have negative SDF values), the mean absolute penetration is used as the loss; otherwise, the loss is zero.

    Args:
        agent_pcs [torch.Tensor]: The normalized point clouds for the agent in the world frame, with shape [B, N, 3].
        sdf_norm_values [torch.Tensor]: The normalized scene SDF values in the world frame, ranging in [-1, 1], 
            with shape [B, grid_resolution, grid_resolution, grid_resolution].

    Return:
        torch.Tensor: The computed collision loss, representing the mean absolute penetration of agent points into 
            obstacles, used for model training. 
    """
    agent_points_num = agent_pcs.shape[1]
    agent_sdf_batch = F.grid_sample(
        sdf_norm_values.unsqueeze(1), 
        agent_pcs[:,:,[2,1,0]].view(-1, agent_points_num, 1, 1, 3),
        padding_mode='border', 
        align_corners=False,
    )
    # if there are no penetrating vertices then set sdf_penetration_loss = 0
    if agent_sdf_batch.lt(0).sum().item() < 1:
        sdf_pene = torch.tensor(0.0, dtype=torch.float32, device=sdf_norm_values.device)
    else:
        sdf_pene = agent_sdf_batch[agent_sdf_batch < 0].abs().mean()
    return sdf_pene
