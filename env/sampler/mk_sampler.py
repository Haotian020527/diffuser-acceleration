import os
import torch
import logging
import trimesh
import numpy as np
from env.agent.mec_kinova import MecKinova
from utils.torch_urdf import TorchURDF
from geometrout.primitive import Sphere
from utils.transform import transform_pointcloud_torch

class MecKinovaSampler:
    """ This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running fk on a subsample
    of the per-link pointclouds that are established at initialization.
    """
    # ignore_link = ["left_inner_finger_pad", "right_inner_finger_pad"]

    def __init__(
        self,
        device,
        num_fixed_points=None,
        use_cache=False,
        max_points=4096,
    ):
        """ Initializes the sampler object, sets the logging level for trimesh, 
        stores the number of fixed points, and calls the internal initialization method.

        Args:
            device [Any]: The device to be used for computation (e.g., 'cpu' or 'cuda').
            num_fixed_points [int, optional]: The number of fixed points to use. Defaults to None.
            use_cache [bool, optional]: Whether to use caching for internal operations. Defaults to False.
            max_points [int, optional]: The maximum number of points allowed. Defaults to 4096.

        Return:
            None
        """
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self._init_internal_(device, use_cache, max_points)

    def _init_internal_(
        self,
        device, 
        use_cache: bool, 
        max_points: int
    ):
        """ Initializes internal data structures for the robot model, loads meshes or creates geometric primitives for each link,
        samples surface points on each link proportional to their surface area, and optionally caches the sampled points for future use.

        Args:
            device [torch.device or str]: The device on which tensors and robot data should be loaded (e.g., 'cpu' or 'cuda').
            use_cache [bool]: Whether to use cached sampled points if available, or generate and cache new ones.
            max_points [int]: The maximum total number of points to sample across all robot links if num_fixed_points is not set.

        Return:
            None. The function sets up internal attributes such as self.points and may save sampled points to a cache file.
        """
        self.device = device
        self.max_points = max_points
        self.robot = TorchURDF.load(
            str(MecKinova.urdf_path), lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.robot.links if len(l.visuals)]
        self.end_effector_links = [l for l in self.links if l.name in MecKinova.END_EFFECTOR_LINK]
        # self.mesh_links = [l for l in self.links if l.visuals[0].geometry.mesh != None]
        if use_cache and self._init_from_cache_(device):
            return

        meshes = []
        for l in self.links: 
            if l.visuals[0].geometry.mesh is None:
                if l.visuals[0].geometry.box:
                    box = l.visuals[0].geometry.box
                    cuboid = trimesh.creation.box(box.size)
                    meshes.append(cuboid)
            else:
                mesh = l.visuals[0].geometry.mesh
                filename = mesh.filename
                scale = (1.0, 1.0, 1.0) if not isinstance(mesh.scale, np.ndarray) else mesh.scale
                filepath = MecKinova.urdf_path.parent / filename
                tmesh = trimesh.load(filepath, force="mesh")
                tmesh.apply_scale(scale)
                meshes.append(tmesh)

        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(
                self.num_fixed_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(max_points * np.array(areas) / np.sum(areas))
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()
            }
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)
    
    def _get_cache_file_name_(self):
        """ Returns the file path for the cached point cloud data based on the number of fixed points.
        If the number of fixed points is specified, returns the corresponding fixed point cloud cache file path.
        Otherwise, returns the full point cloud cache file path.

        Args:
            self: Instance of the class containing configuration and cache directory information.

        Return:
            Path: The file path to the cached point cloud data as a Path object.
        """
        if self.num_fixed_points is not None:
            return (
                MecKinova.pointcloud_cache
                / f"fixed_point_cloud_{self.num_fixed_points}.npy"
            )
        else:
            return MecKinova.pointcloud_cache / "full_point_cloud.npy"

    def _init_from_cache_(self, device):
        """ Loads cached point cloud data from a file if it exists, and initializes the `self.points` attribute
        with the loaded data as PyTorch tensors on the specified device.

        Args:
            device [torch.device or str]: The device on which the loaded tensors should be allocated.

        Return:
            bool: Returns True if the cache file exists and the data is successfully loaded; otherwise, returns False.
        """
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in points.item().items()
        }
        return True
    
    def sample(self, config, num_points=None):
        """ Samples points from the surface of the robot by calling fk.

        Args:
            config [Tensor]: Tensor of shape (M,) or (N, M), where M is the number of actuated joints. Represents the robot configuration(s).
            num_points [int, optional]: Number of points to sample from the robot surface. If None, uses the default number of fixed points.

        Return:
            Tensor: Returns a point cloud of shape (N, num_points, 3) representing sampled points on the robot surface.
        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud_torch(
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_base(self, config, num_points=None):
        """ Samples points from the base surface of the robot by calling fk.

        Args:
            config [Tensor]: Tensor of shape (M,) or (N, M), where M is the number of actuated joints. Represents the robot configuration(s).
            num_points [int, optional]: Number of points to sample from the robot base. If None, all points are returned.

        Return:
            Tensor: A point cloud of shape (N, num_points, 3) representing sampled points from the robot base surface.
        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name in MecKinova.BASE_LINK:
                fk_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.points[l.name]
                    .float()
                    .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                    fk_transforms[l.name],
                    in_place=True,
                )
                fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_arm(self, config, num_points=None):
        """ Samples points from the arm surface of the robot by calling fk.

        Args:
            config [Tensor]: A tensor of shape (M,) or (N, M), where M is the number of actuated joints. Represents one or more joint configurations.
            num_points [int, optional]: The number of points to sample from the arm surface for each configuration. If None, all available points are returned.
        Return:
            Tensor: A tensor of shape (N, num_points, 3) representing the sampled point cloud(s) of the robot arm surface for each configuration.

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name in MecKinova.ARM_LINK:
                fk_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.points[l.name]
                    .float()
                    .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                    fk_transforms[l.name],
                    in_place=True,
                )
                fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_gripper(self, config, num_points=None):
        """ Samples points from the robot gripper's surface by performing forward kinematics (FK) 
        and transforming cached points to the world frame. This function is similar to 
        `sample_end_effector` but uses cached gripper points and applies coordinate transformations.

        Args:
            config [Tensor]: A tensor of shape (M,) or (N, M), where M is the number of actuated joints. Represents the robot's joint configuration(s).
            num_points [int, optional]: The number of gripper surface points to sample. If None, all available points are returned.

        Return:
            Tensor: A tensor of shape (N, num_points, 3) representing the sampled 3D point cloud of the robot gripper for each configuration.
        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None) 
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name in MecKinova.END_EFFECTOR_LINK:
                fk_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.points[l.name]
                    .float()
                    .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                    fk_transforms[l.name],
                    in_place=True,
                )
                fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def end_effector_pose(self, config, frame="end_effector_link") -> torch.Tensor:
        """ Computes and returns the pose of the specified end effector frame for the given robot configuration(s).
        If a single configuration is provided, it is reshaped to a batch of one for processing.

        Args:
            config [torch.Tensor]: The robot joint configuration(s), either as a 1D tensor (single configuration) or 2D tensor (batch of configurations).
            frame [str]: The name of the end effector frame for which the pose is to be computed. Defaults to "end_effector_link".

        Return:
            torch.Tensor: The pose(s) of the specified end effector frame corresponding to the input configuration(s).
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        return fk[frame]
    
    def _get_eef_cache_file_name_(self, eef_points_num):
        """ Generates the cache file path for the end-effector (EEF) point cloud based on the specified number of EEF points.

        Args:
            eef_points_num [int]: The number of points in the EEF point cloud.

        Return:
            Path object representing the file path to the cached EEF point cloud .npy file.
        """
        return (
            MecKinova.pointcloud_cache
            / f"eef_point_cloud_{eef_points_num}.npy"
        )

    def _init_from_eef_cache_(self, device, eef_points_num):
        """ Initializes the end-effector (EEF) points from a cached file if it exists. 
        Loads the cached EEF points and converts them into PyTorch tensors on the specified device. 
        If the cache file does not exist, the function returns False.

        Args:
            device [torch.device or str]: The device on which the tensors should be allocated (e.g., 'cpu' or 'cuda').
            eef_points_num [int]: The number of EEF points, used to determine the cache file name.

        Return:
            bool: Returns True if the EEF points were successfully loaded from the cache; otherwise, returns False.
        """
        eef_file_name = self._get_eef_cache_file_name_(eef_points_num)
        if not os.path.exists(eef_file_name):
            return False
        eef_points = np.load(
            eef_file_name,
            allow_pickle=True,
        )
        self.eef_points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in eef_points.item().items()
        }
        return True
    
    def sample_end_effector(self, config, sample_points=512, use_cache=False):
        """ Samples point clouds from the end effector meshes or boxes of a robot, 
        transforms them according to the given configuration, and returns the transformed point clouds. 
        Optionally uses or saves a cache of sampled points for efficiency.

        Args:
            config [torch.Tensor]: The robot configuration(s) for which to compute the forward kinematics and transform the sampled points. 
            Shape can be (n,) or (batch_size, n).
            sample_points [int]: Number of points to sample from the end effector surfaces. Default is 512.
            use_cache [bool]: Whether to use cached sampled points if available, and save new samples to cache. Default is False.
        Return:
            torch.Tensor: The transformed point clouds of the end effector(s), concatenated along the point dimension. Shape is (batch_size, sample_points, 3).
        """
        self.eef_points = {}
        if use_cache and self._init_from_eef_cache_(self.device, sample_points):
            pass
        else:
            end_effector_meshes = []
            for l in self.end_effector_links: 
                if l.visuals[0].geometry.mesh is None:
                    if l.visuals[0].geometry.box:
                        box = l.visuals[0].geometry.box
                        cuboid = trimesh.creation.box(box.size)
                        end_effector_meshes.append(cuboid)
                else:
                    mesh = l.visuals[0].geometry.mesh
                    filename = mesh.filename
                    scale = (1.0, 1.0, 1.0) if not isinstance(mesh.scale, np.ndarray) else mesh.scale
                    filepath = MecKinova.urdf_path.parent / filename
                    tmesh = trimesh.load(filepath, force="mesh")
                    tmesh.apply_scale(scale)
                    end_effector_meshes.append(tmesh)
            
            areas = [mesh.bounding_box_oriented.area for mesh in end_effector_meshes]

            num_points = np.round(
                sample_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += sample_points - np.sum(num_points)
            assert np.sum(num_points) == sample_points

            # sample Points
            for i in range(len(end_effector_meshes)):
                pc = trimesh.sample.sample_surface(end_effector_meshes[i], int(num_points[i]))[0]
                self.eef_points[self.end_effector_links[i].name] = torch.as_tensor(
                    pc, device=self.device
                ).unsqueeze(0)

            # save points
            if use_cache:
                points_to_save = {
                    k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.eef_points.items()
                }
                eef_file_name = self._get_eef_cache_file_name_(sample_points)
                print(f"Saving new file to cache: {eef_file_name}")
                np.save(eef_file_name, points_to_save)
        
        # transform
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = config
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())

        assert len(self.links) == len(values)
        ef_transforms = {}
        ef_points = []
        for idx, l in enumerate(self.links):
            if l in self.end_effector_links:
                ef_transforms[l.name] = values[idx]
                pc = transform_pointcloud_torch(
                    self.eef_points[l.name]
                    .float()
                    .repeat((ef_transforms[l.name].shape[0], 1, 1)),
                    ef_transforms[l.name],
                    in_place=True,
                )
                ef_points.append(pc)
        pc = torch.cat(ef_points, dim=1)
        return pc


class MecKinovaCollisionSampler:
    """MecKinovaCollisionSampler is a utility class for sampling and computing collision-related geometric representations
    for a Kinova robot model using sphere approximations. It loads the robot's URDF, constructs sets of spheres to
    approximate the robot's links, and provides methods to sample surface points from these spheres and compute their
    transformed positions given a robot configuration.
    """
    def __init__(
        self,
        device,
        margin=0.0,
    ):
        """ Initializes the sampler for the MecKinova robot, loading its URDF model, 
        configuring collision spheres with optional margin, and precomputing 
        surface sample points for each robot link.

        Args:
            device [torch.device or str]: The device on which tensors and models will be loaded (e.g., 'cpu' or 'cuda').
            margin [float, optional]: Additional margin to add to the radius of each collision sphere. Default is 0.0.

        Return:
            None. Initializes class attributes including the robot model, collision spheres, and sampled surface points.
        """
        logging.getLogger("trimesh").setLevel("ERROR")
        self.robot = TorchURDF.load(
            str(MecKinova.urdf_path), lazy_load_meshes=True, device=device
        )
        self.spheres = []
        for radius, point_set in MecKinova.SPHERES:
            sphere_centers = {
                k: torch.as_tensor(v).to(device) for k, v in point_set.items()
            }
            if not len(sphere_centers):
                continue
            self.spheres.append(
                (
                    radius + margin,
                    sphere_centers,
                )
            )
        
        all_spheres = {}
        for radius, point_set in MecKinova.SPHERES:
            for link_name, centers in point_set.items():
                for c in centers:
                    all_spheres[link_name] = all_spheres.get(link_name, []) + [
                        Sphere(np.asarray(c), radius + margin)
                    ]
        
        total_points = 10000
        surface_scalar_sum = sum(
            [sum([s.radius ** 2 for s in v]) for v in all_spheres.values()]
        )
        surface_scalar = total_points / surface_scalar_sum
        self.link_points = {}
        for link_name, spheres in all_spheres.items():
            self.link_points[link_name] = torch.as_tensor(
                np.concatenate(
                    [
                        s.sample_surface(int(surface_scalar * s.radius ** 2))
                        for s in spheres
                    ],
                    axis=0,
                ),
                device=device,
            )
    
    def sample(self, config, n):
        """ Generates a point cloud by sampling points from the robot's links given a configuration.
        The function first ensures the configuration tensor has the correct shape, concatenates
        default prismatic values, computes forward kinematics for each link, transforms the
        predefined link points, and finally samples 'n' points from the resulting point cloud.

        Args:
            config [torch.Tensor]: The input configuration tensor for the robot. Shape: (num_joints,) or (batch_size, num_joints).
            n [int]: The number of points to sample from the generated point cloud.

        Return:
            torch.Tensor: A tensor containing 'n' sampled points from the robot's link point cloud. Shape: (batch_size, n, 3).
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        pointcloud = []
        for link_name, points in self.link_points.items():
            pc = transform_pointcloud_torch(
                points.float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name],
                in_place=True,
            )
            pointcloud.append(pc)
        pc = torch.cat(pointcloud, dim=1)
        return pc[:, np.random.choice(pc.shape[1], n, replace=False), :]

    def compute_spheres(self, config):
        """ Computes the transformed positions of predefined spheres attached to robot links for a given configuration.
        The function processes each sphere group, applies the corresponding forward kinematics transformation, and returns the transformed points.

        Args:
            config [torch.Tensor]: The robot configuration tensor. Should be of shape (n, d) or (d,) where n is the batch size and d is the configuration dimension.

        Return:
            List[Tuple[float, torch.Tensor]]: A list of tuples, each containing a sphere radius and a tensor of transformed sphere points for each 
            configuration in the batch.
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg  = config
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                pc = transform_pointcloud_torch(
                    spheres[link_name]
                    .type_as(cfg)
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name].type_as(cfg),
                    in_place=True,
                )
                fk_points.append(pc)
            points.append((radius, torch.cat(fk_points, dim=1)))
        return points