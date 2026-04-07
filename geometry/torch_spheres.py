# Note: original code can be found on Github at
# https://github.com/NVlabs/motion-policy-networks/blob/main/mpinets/geometry.py

import numpy as np
import torch

class TorchSpheres:
    """ A Pytorch representation of a batch of M spheres (i.e. B elements in the batch,
    M spheres per element). Any of these spheres can have zero volume (these
    will be masked out during calculation of the various functions in this
    class, such as sdf).
    """

    def __init__(self, centers: torch.Tensor, radii: torch.Tensor):
        """
        :param centers torch.Tensor: a set of centers, has dim [B, M, 3]
        :param radii torch.Tensor: a set of radii, has dim [B, M, 1]
        """
        assert centers.ndim == 3
        assert radii.ndim == 3

        # TODO It would be more memory efficient to rely more heavily on broadcasting
        # in some cases where multiple spheres have the same size
        assert centers.ndim == radii.ndim

        # This logic is to determine the batch size. Either batch sizes need to
        # match or if only one variable has a batch size, that one is assumed to
        # be the batch size

        self.centers = centers
        self.radii = radii
        self.mask = ~torch.isclose(self.radii, torch.zeros(1).type_as(centers)).squeeze(
            -1
        )

    def surface_area(self) -> torch.Tensor:
        """
        Calculates the surface area of the spheres

        :rtype torch.Tensor: A tensor of the surface areas of the spheres
        """
        area = 4 * np.pi * torch.pow(self.radii, 3)
        return area

    def sample_surface(self, num_points: int) -> torch.Tensor:
        """
        Samples points from all spheres, including ones with zero volume

        :param num_points int: The number of points to sample per sphere
        :rtype torch.Tensor: The points, has dim [B, M, N]
        """
        B, M, _ = self.centers.shape
        unnormalized_points = torch.rand((B, M, num_points, 3))
        normalized = (
            unnormalized_points
            / torch.linalg.norm(unnormalized_points, dim=-1)[:, :, :, None]
        )
        random_points = (
            normalized * self.radii[:, :, None, :] + self.centers[:, :, None, :]
        )
        return random_points

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        :param points torch.Tensor: The points with which to calculate the
                                    SDF, has dim [B, N, 3] (N is the number of points)
        :rtype torch.Tensor: The scene SDF value for each point (i.e. the minimum SDF
                             value for each of the M spheres), has dim [B, N]
        """
        assert points.ndim == 3
        B, M, _ = self.radii.shape
        _, N, _ = points.shape
        all_sdfs = float("inf") * torch.ones(B, M, N).type_as(points)
        distances = points[:, None, :, :] - self.centers[:, :, None, :]
        all_sdfs[self.mask] = (
            torch.linalg.norm(distances[self.mask], dim=-1) - self.radii[self.mask]
        )
        return torch.min(all_sdfs, dim=1)[0]

    def sdf_sequence(self, points: torch.Tensor) -> torch.Tensor:
        """
        Calculates SDF values for a time sequence of point clouds
        :param points torch.Tensor: The batched sequence of point clouds with
                                    dimension [B, T, N, 3] (T in sequence length,
                                    N is number of points)
        :rtype torch.Tensor: The scene SDF for each point at each timestep (i.e. the minimum
                             SDF value across the M spheres at each timestep),
                             has dim [B, T, N]
        """
        assert points.ndim == 4
        B, M, _ = self.radii.shape
        _, T, N, _ = points.shape
        all_sdfs = float("inf") * torch.ones(B, M, T, N).type_as(points)
        distances = points[:, None, :, :] - self.centers[:, :, None, None, :]
        all_sdfs[self.mask] = (
            torch.linalg.norm(distances[self.mask], dim=-1)
            - self.radii[:, :, None, :][self.mask]
        )
        return torch.min(all_sdfs, dim=1)[0]