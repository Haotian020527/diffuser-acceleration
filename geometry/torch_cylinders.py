# Note: original code can be found on Github at
# https://github.com/NVlabs/motion-policy-networks/blob/main/mpinets/geometry.py

import torch
from geometrout.primitive import Cylinder

class TorchCylinders:
    """ A Pytorch representation of a batch of M cylinders (i.e. B elements in the batch,
    M cylinders per element). Any of these cylinders can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(
        self,
        centers: torch.Tensor,
        radii: torch.Tensor,
        heights: torch.Tensor,
        quaternions: torch.Tensor,
    ):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param radii torch.Tensor: Has dim [B, M, 1]
        :param heights torch.Tensor: Has dim [B, M, 1]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as (w, x, y, z)
        """
        assert centers.ndim == 3
        assert radii.ndim == 3
        assert heights.ndim == 3
        assert quaternions.ndim == 3

        self.radii = radii
        self.heights = heights
        self.centers = centers

        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]
        self._init_frames()
        # Mask for nonzero volumes
        self.mask = ~torch.logical_or(
            torch.isclose(self.radii, torch.zeros(1).type_as(centers)).squeeze(-1),
            torch.isclose(self.heights, torch.zeros(1).type_as(centers)).squeeze(-1),
        )

    def geometrout(self):
        """
        Helper method to convert this into geometrout primitives
        """
        B, M, _ = self.centers.shape
        return [
            [
                Cylinder(
                    center=self.centers[bidx, midx, :].detach().cpu().numpy(),
                    radius=self.radii[bidx, midx, 0].detach().cpu().numpy(),
                    height=self.heights[bidx, midx, 0].detach().cpu().numpy(),
                    quaternion=self.quats[bidx, midx, :].detach().cpu().numpy(),
                )
                for midx in range(M)
                if self.mask[bidx, midx]
            ]
            for bidx in range(B)
        ]

    def _init_frames(self):
        """
        In order to calculate the SDF, we need to calculate the inverse
        transformation of the cylinder. This is because we are transforming
        points in the world frame into the cylinder frame.
        """
        # Initialize the inverse rotation
        w = self.quats[:, :, 0]
        x = -self.quats[:, :, 1]
        y = -self.quats[:, :, 2]
        z = -self.quats[:, :, 3]

        # Naming is a little disingenuous here because I'm multiplying everything by two,
        # but can't put numbers at the beginning of variable names.
        xx = 2 * torch.pow(x, 2)
        yy = 2 * torch.pow(y, 2)
        zz = 2 * torch.pow(z, 2)

        wx = 2 * w * x
        wy = 2 * w * y
        wz = 2 * w * z
        xy = 2 * x * y
        xz = 2 * x * z
        yz = 2 * y * z

        B, M, _ = self.centers.shape
        B = self.centers.size(0)
        M = self.centers.size(1)
        self.inv_frames = torch.zeros((B, M, 4, 4)).type_as(self.centers)
        self.inv_frames[:, :, 3, 3] = 1

        R = torch.stack(
            [
                torch.stack([1 - yy - zz, xy - wz, xz + wy], dim=2),
                torch.stack([xy + wz, 1 - xx - zz, yz - wx], dim=2),
                torch.stack([xz - wy, yz - wx, 1 - xx - yy], dim=2),
            ],
            dim=2,
        )
        Rt = torch.matmul(R, -1 * self.centers.unsqueeze(3)).squeeze(3)

        # Fill in the rotation matrices
        self.inv_frames[:, :, :3, :3] = R

        # Invert the transform by multiplying the inverse translation by the inverse rotation
        self.inv_frames[:, :, :3, 3] = Rt

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the SDF, has
                                    dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M cylinders), has dim [B, N]
        """
        assert points.ndim == 3
        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.size(0), points.size(1)).type_as(
                points
            )

        homog_points = torch.cat(
            (
                points,
                torch.ones((points.size(0), points.size(1), 1)).type_as(points),
            ),
            dim=2,
        )
        # Next, project these points into their respective cylinder frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, :, :], homog_points[:, None, :, :, None]
        ).squeeze(-1)[:, :, :, :3]
        B, M, N, _ = points_proj.shape
        masked_points = points_proj[self.mask]

        surface_distance_xy = torch.linalg.norm(masked_points[:, :, :2], dim=2)
        z_distance = masked_points[:, :, 2]

        half_extents_2d = torch.stack(
            (self.radii[self.mask], self.heights[self.mask] / 2), dim=2
        )
        points_2d = torch.stack((surface_distance_xy, z_distance), dim=2)
        distances_2d = torch.abs(points_2d) - half_extents_2d

        # This is distance only for points outside the box, all points inside return zero
        outside = torch.linalg.norm(
            torch.maximum(distances_2d, torch.zeros_like(distances_2d)), dim=2
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance_2d = torch.max(distances_2d, dim=2).values
        inside = torch.minimum(
            inner_max_distance_2d, torch.zeros_like(inner_max_distance_2d)
        )

        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep
                             (i.e. the minimum SDF value across the M cylinders at
                             each timestep), has dim [B, T, N]
        """
        assert points.ndim == 4

        # We are going to need to map points in the global frame to the cylinder frame
        # First take the points and make them homogeneous by adding a one to the end

        assert points.size(0) == self.centers.size(0)
        if torch.all(~self.mask):
            return float("inf") * torch.ones(points.shape[:-1]).type_as(points)

        homog_points = torch.cat(
            (
                points,
                torch.ones((*points.shape[:-1], 1)).type_as(points),
            ),
            dim=3,
        )
        # Next, project these points into their respective cylinder frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, None, :, :],
            homog_points[:, None, :, :, :, None],
        ).squeeze(-1)[:, :, :, :, :3]
        B, M, T, N, _ = points_proj.shape
        assert T == points.size(1)
        assert N == points.size(2)
        masked_points = points_proj[self.mask]

        surface_distance_xy = torch.linalg.norm(masked_points[:, :, :, :2], dim=-1)
        z_distance = masked_points[:, :, :, 2]

        half_extents_2d = torch.stack(
            (self.radii[self.mask], self.heights[self.mask] / 2), dim=2
        )[:, :, None, :]
        points_2d = torch.stack((surface_distance_xy, z_distance), dim=3)
        distances_2d = torch.abs(points_2d) - half_extents_2d

        # This is distance only for points outside the box, all points inside return zero
        outside = torch.linalg.norm(
            torch.maximum(distances_2d, torch.zeros_like(distances_2d)), dim=3
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance_2d = torch.max(distances_2d, dim=3).values
        inside = torch.minimum(
            inner_max_distance_2d, torch.zeros_like(inner_max_distance_2d)
        )

        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]