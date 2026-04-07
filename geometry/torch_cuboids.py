# Note: original code can be found on Github at
# https://github.com/NVlabs/motion-policy-networks/blob/main/mpinets/geometry.py

import torch
from geometrout.primitive import Cuboid
from cprint import *

class TorchCuboids:
    """ A Pytorch representation of a batch of M cuboids (i.e. B elements in the batch, M cuboids per element).
    Any of these cuboids can have zero volume (these will be masked out during calculation of the various 
    functions in this class, such as sdf).
    """

    def __init__(
        self, centers: torch.Tensor, dims: torch.Tensor, quaternions: torch.Tensor
    ):
        """
        :param centers torch.Tensor: Has dim [B, M, 3]
        :param dims torch.Tensor: Has dim [B, M, 3]
        :param quaternions torch.Tensor: Has dim [B, M, 4] with quaternions formatted as w, x, y, z
        """
        assert centers.ndim == 3
        assert dims.ndim == 3
        assert quaternions.ndim == 3

        self.dims = dims
        self.centers = centers

        # It's helpful to ensure the quaternions are normalized
        self.quats = quaternions / torch.linalg.norm(quaternions, dim=2)[:, :, None]

        self._init_frames()

        # Mask for nonzero volumes
        self.mask = ~torch.any(
            torch.isclose(self.dims, torch.zeros(1).type_as(centers)), dim=-1
        )

    def geometrout(self):
        """
        Helper method to convert this into geometrout primitives
        """
        B, M, _ = self.centers.shape
        return [
            [
                Cuboid(
                    center=self.centers[bidx, midx, :].detach().cpu().numpy(),
                    dims=self.dims[bidx, midx, :].detach().cpu().numpy(),
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
        transformation of the cuboid. This is because we are transforming points
        in the world frame into the cuboid frame.
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

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the cuboids

        :rtype torch.Tensor: A tensor of the surface areas of the cuboids
        """
        area = 2 * (
            self.dims[:, :, 0] * self.dims[:, :, 1]
            + self.dims[:, :, 0] * self.dims[:, :, 2]
            + self.dims[:, :, 1] * self.dims[:, :, 2]
        )
        return area

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the SDF, has
                                    dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                                value for each of the M cuboids), has dim [B, N]
        """
        assert points.ndim == 3 
        # We are going to need to map points in the global frame to the cuboid frame
        # First take the points and make them homogeneous by adding a one to the end

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
        # Next, project these points into their respective cuboid frames. Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, :, :], homog_points[:, None, :, :, None]
        ).squeeze(-1)[:, :, :, :3]

        B, M, N, _ = points_proj.shape 

        masked_points = points_proj[self.mask]

        # The follow computations are adapted from here
        # https://github.com/fogleman/sdf/blob/main/sdf/d3.py
        # Move points to top corner

        distances = torch.abs(masked_points) - (self.dims[self.mask] / 2)[:, None, :]

        # This is distance only for points outside the box, all points inside return zero
        # This probably needs to be fixed or else there will be a nan gradient

        outside = torch.linalg.norm(
            torch.maximum(distances, torch.zeros_like(distances)), dim=-1
        ) 

        # This is distance for points inside the box, all others return zero
        inner_max_distance = torch.max(distances, dim=-1).values

        inside = torch.minimum(inner_max_distance, torch.zeros_like(inner_max_distance))

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
                             (i.e. the minimum SDF value across the M cuboids
                             at each timestep), has dim [B, T, N]
        """
        assert points.ndim == 4

        # We are going to need to map points in the global frame to the cuboid frame
        # First take the points and make them homogeneous by adding a one to the end

        # points_from_volumes = points[self.nonzero_volumes, :, :]
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
        # Next, project these points into their respective cuboid frames
        # Will return [B x M x N x 3]

        points_proj = torch.matmul(
            self.inv_frames[:, :, None, None, :, :],
            homog_points[:, None, :, :, :, None],
        ).squeeze(-1)[:, :, :, :, :3]
        B, M, T, N, _ = points_proj.shape
        assert T == points.size(1)
        assert N == points.size(2)
        masked_points = points_proj[self.mask]

        # The follow computations are adapted from here
        # https://github.com/fogleman/sdf/blob/main/sdf/d3.py
        # Move points to top corner

        distances = (
            torch.abs(masked_points) - (self.dims[self.mask] / 2)[:, None, None, :]
        )
        # This is distance only for points outside the box, all points inside return zero
        # This probably needs to be fixed or else there will be a nan gradient

        outside = torch.linalg.norm(
            torch.maximum(distances, torch.zeros_like(distances)), dim=-1
        )
        # This is distance for points inside the box, all others return zero
        inner_max_distance = torch.max(distances, dim=-1).values
        inside = torch.minimum(inner_max_distance, torch.zeros_like(inner_max_distance))
        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        all_sdfs[self.mask] = outside + inside
        return torch.min(all_sdfs, dim=1)[0]