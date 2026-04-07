import os
import re
import copy
import trimesh
import numpy as np
from lxml import etree
from pathlib import Path
from typing import List, Optional, Union
from yourdfpy import URDF, Link
from utils.transform import TransformationMatrix2QuaternionXYZ
from utils.path import RootPath

class Scene():
    """ A high-level class for managing and manipulating 3D scenes described by URDF files. 
    This class provides utilities for loading, transforming, exporting, and sampling from 
    scene representations, including collision and visual models. It supports operations 
    such as cropping the scene, sampling point clouds, updating object poses, and extracting 
    mesh data for specific links. The class also handles the creation of modified URDFs 
    (e.g., without flooring or ceiling) for different purposes.
    """

    def __init__(self, name: str) -> None:
        """ Initializes the Scene object with the given name.

        Args:
            name [str]: The name of the scene, used to construct file paths for various scene resources.

        Return:
            None. Initializes various file paths and loads URDF models for the scene.
        """
        self.name = name
        self.scene_dir = self.resolve_scene_dir(name)
        self._urdf_path = str(self.resolve_scene_urdf_path(name, scene_dir=self.scene_dir))
        self.urdf_dir = Path(self._urdf_path).parent
        # Keep generated URDFs next to the source URDF so relative mesh paths remain valid.
        self._urdf_collision_path = str(self.urdf_dir / "main_without_flooring.urdf")
        self._urdf_visual_path = str(self.urdf_dir / "main_without_ceiling.urdf")
        self._create_urdf_without_flooring()
        self._create_urdf_without_ceiling()

        self._attachment_path = str(self.scene_dir / "attachments.json")
        self._grasp_objects_positions = str(self.scene_dir / "grasp_objects_positions.json")
        self._sdf_path = str(self.scene_dir / "sdf.npy")
        self.grasping_poses_dir = self.scene_dir / "grasping_poses"
        self.urdf = URDF.load(self._urdf_path)
        self.urdf_collision = URDF.load(self._urdf_collision_path)
        self.urdf_visual = URDF.load(self._urdf_visual_path)

    @staticmethod
    def _scene_name_candidates(scene_name: str) -> List[str]:
        """Generate robust candidates for scene directory lookup."""
        name = str(scene_name).strip()
        candidates = [name]

        # Remove extension if a filename accidentally appears in dataset.
        if name.endswith(".urdf"):
            candidates.append(Path(name).stem)

        # Numeric aliases: 6339 <-> physcene_6339
        digits = "".join(re.findall(r"\d+", name))
        if digits:
            candidates.extend([digits, f"physcene_{digits}", f"scene_{digits}"])

        # De-duplicate while preserving order.
        uniq: List[str] = []
        for c in candidates:
            if c not in uniq:
                uniq.append(c)
        return uniq

    @staticmethod
    def resolve_scene_dir(scene_name: str) -> Path:
        """Resolve scene directory from dataset scene name and current asset layout."""
        scene_root = RootPath.SCENE
        candidates = Scene._scene_name_candidates(scene_name)

        for c in candidates:
            p = scene_root / c
            if p.is_dir():
                return p

        # Fallback: recursive search for directory with exact candidate name.
        for c in candidates:
            matches = list(scene_root.rglob(c))
            for m in matches:
                if m.is_dir():
                    return m

        # If not found, provide actionable diagnostic.
        root_exists = scene_root.exists()
        few_dirs: List[str] = []
        if root_exists:
            few_dirs = [p.name for p in sorted(scene_root.iterdir()) if p.is_dir()][:10]
        raise FileNotFoundError(
            f"Scene directory not found for scene_name='{scene_name}'. "
            f"scene_root='{scene_root}', root_exists={root_exists}, "
            f"tried={candidates}, sample_dirs={few_dirs}"
        )

    @staticmethod
    def resolve_scene_urdf_path(scene_name: str, scene_dir: Optional[Path] = None) -> Path:
        """Resolve main URDF path under a scene directory with fallback filenames."""
        if scene_dir is None:
            scene_dir = Scene.resolve_scene_dir(scene_name)

        preferred = [
            scene_dir / "main.urdf",
            scene_dir / "main_simu.urdf",
        ]
        for p in preferred:
            if p.is_file():
                return p

        # Fallback: recurse, because some assets are nested like
        # scene_dir/<scene_name>/main.urdf
        nested_preferred = []
        for filename in ("main.urdf", "main_simu.urdf"):
            nested_preferred.extend(sorted(scene_dir.rglob(filename)))
        if len(nested_preferred) > 0:
            nested_preferred = [p for p in nested_preferred if p.is_file()]
            nested_preferred.sort(key=lambda p: len(str(p)))
            return nested_preferred[0]

        # Last fallback: pick any urdf in (possibly nested) directory.
        urdfs = sorted([p for p in scene_dir.rglob("*.urdf") if p.is_file()])
        if len(urdfs) > 0:
            urdfs.sort(key=lambda p: len(str(p)))
            return urdfs[0]

        few_children = [p.name for p in sorted(scene_dir.iterdir())[:15]] if scene_dir.exists() else []
        raise FileNotFoundError(
            f"No URDF file found under scene_dir='{scene_dir}' for scene_name='{scene_name}'. "
            f"children_sample={few_children}"
        )

    def _resolve_link_mesh_path(self, link_filename: str) -> str:
        """Resolve mesh path robustly for multiple URDF packaging styles."""
        if os.path.isabs(link_filename) and os.path.isfile(link_filename):
            return link_filename

        cleaned = link_filename.replace("package://", "").lstrip("/\\")
        candidates = [
            self.urdf_dir / link_filename,
            self.scene_dir / link_filename,
            self.urdf_dir / cleaned,
            self.scene_dir / cleaned,
        ]
        for c in candidates:
            c = c.resolve()
            if c.is_file():
                return str(c)

        # Fallback by basename lookup under scene_dir.
        basename = Path(cleaned).name
        if basename:
            matches = [p for p in self.scene_dir.rglob(basename) if p.is_file()]
            if len(matches) > 0:
                return str(matches[0])

        raise FileNotFoundError(
            f"Cannot resolve mesh file '{link_filename}' for scene='{self.name}'. "
            f"urdf_path='{self._urdf_path}', scene_dir='{self.scene_dir}'"
        )

    @property
    def trimesh_original(self):
        """ This property retrieves the original trimesh representation of the scene by dumping the current URDF scene 
        and processing it through the `_simple_trimesh` method.

        Args:
            None

        Return:
            trimesh.Trimesh: The simplified trimesh object representing the original scene.
        """
        self._scene = self.urdf.scene
        self._trimesh_original = self._scene.dump(concatenate=True)
        return self._simple_trimesh(self._trimesh_original)
    
    @property
    def trimesh_collision(self):
        """ Returns a simplified trimesh representation of the collision geometry for the scene.
        It updates the internal scene collision and trimesh collision attributes before processing.

        Args:
            None

        Return:
            The simplified trimesh object representing the scene's collision geometry.
        """
        self._scene_collision = self.urdf_collision.scene
        self._trimesh_collision = self._scene_collision.dump(concatenate=True)
        return self._simple_trimesh(self._trimesh_collision)
    
    @property
    def trimesh_visual(self):
        """ This property retrieves the visual representation of the scene as a single trimesh object.
        It first obtains the scene visual from the URDF visual, dumps it into a concatenated trimesh,
        and then processes it using the `_simple_trimesh` method.

        Args:
            None

        Return:
            trimesh.Trimesh: The processed trimesh object representing the scene's visual geometry.
        """
        self._scene_visual = self.urdf_visual.scene
        self._trimesh_visual = self._scene_visual.dump(concatenate=True)
        return self._simple_trimesh(self._trimesh_visual)
    
    @property
    def trimesh_original_vis(self):
        """ This property retrieves the original trimesh representations from the URDF scene.
        It first updates the internal scene reference, then dumps the scene into its constituent
        trimesh objects without concatenating them.

        Args:
            None

        Return:
            list: The list of original trimesh objects from the scene.
        """
        self._scene = self.urdf.scene
        self._trimesh_original = self._scene.dump(concatenate=False)
        return self._trimesh_original
    
    @property
    def trimesh_collision_vis(self):
        """ This property retrieves the collision meshes from the URDF collision scene,
        dumps them as a list of trimesh objects without concatenation, and returns them.

        Args:
            None

        Return:
            list: A list of trimesh objects representing the collision geometry of the scene.
        """
        self._scene_collision = self.urdf_collision.scene
        self._trimesh_collision = self._scene_collision.dump(concatenate=False)
        return self._trimesh_collision
    
    @property
    def trimesh_visual_vis(self):
        """ This property retrieves the visual representation of the scene from the URDF visual object,
        dumps it into a list of trimesh visual objects without concatenation, and returns the result.

        Args:
            None

        Return:
            list: A list of trimesh visual objects representing the scene's visual geometry.
        """
        self._scene_visual = self.urdf_visual.scene
        self._trimesh_visual = self._scene_visual.dump(concatenate=False)
        return self._trimesh_visual
    
    def transform(self, matrix: Union[List, np.ndarray]):
        """ Applies a homogeneous transformation matrix to the URDF scene, collision scene, and visual scene.

        Args:
            matrix [Union[List, np.ndarray]]: A (4, 4) float homogeneous transformation matrix to be applied.

        Return:
            None. The function updates the transformation of the associated scenes in place.
        """
        self.urdf.scene.apply_transform(np.asarray(matrix))
        self.urdf_collision.scene.apply_transform(np.asarray(matrix))
        self.urdf_visual.scene.apply_transform(np.asarray(matrix))
    
    def translation(self, vector: Union[List, np.ndarray]):
        """ Applies a translation to the scene in the XYZ directions using the provided vector.

        Args:
            vector [Union[List, np.ndarray]]: A 3-element vector specifying the translation in the X, Y, and Z axes.

        Return:
            None. The function updates the translation of the scene objects in place.
        """
        self.urdf.scene.apply_translation(np.asarray(vector))
        self.urdf_collision.scene.apply_translation(np.asarray(vector))
        self.urdf_visual.scene.apply_translation(np.asarray(vector))
    
    def export(self, path: str):
        """ Exports the current URDF scene to a specified file path.

        Args:
            path [str]: The file path where the scene will be exported.

        Return:
            None. The function exports the current scene to the specified file path.
        """
        self.urdf.scene.export(path)

    def export_visual(self, path: str):
        """ Exports the visual representation of the scene to the specified file path.

        Args:
            path [str]: The file path where the visual scene will be exported.

        Return:
            None. The function saves the visual scene to the given path.
        """
        self.urdf_visual.scene.export(path)

    def export_collision(self, path: str):
        """ Exports the collision scene to a specified file path using the URDF collision exporter.

        Args:
            path [str]: The file path where the collision scene will be exported.

        Return:
            None: This function does not return a value.
        """
        self.urdf_collision.scene.export(path)

    def update_object_position_by_transformation_matrix(
        self,  
        attach_link_name: str,
        transformation_matrix: Union[List, np.ndarray]
    ) -> None:
        """ Updates the pose of the specified attach_link in the scene by applying a transformation matrix.
        This method synchronizes the transformation across the main, collision, and visual URDF scenes.

        Args:
            attach_link_name [str]: The name of the attach_link whose pose will be updated.
            transformation_matrix [Union[List, np.ndarray]]: The transformation matrix representing the pose from "world" to "attach_link".

        Return:
            None: This function does not return a value. It updates the internal state of the scene.
        """
        self.urdf.scene.graph.update(
            frame_to=attach_link_name, 
            frame_from="world", 
            matrix=np.asarray(transformation_matrix)
        )
        self.urdf_collision.scene.graph.update(
            frame_to=attach_link_name, 
            frame_from="world", 
            matrix=np.asarray(transformation_matrix)
        )
        self.urdf_visual.scene.graph.update(
            frame_to=attach_link_name, 
            frame_from="world", 
            matrix=np.asarray(transformation_matrix)
        )

    def return_init_state(self):
        """ Initializes the URDF collision and visual models by loading them from the specified file paths.

        Args:
            self: Instance of the class containing the URDF file paths as attributes.

        Return:
            None. The function sets the urdf_collision and urdf_visual attributes of the instance.
        """
        self.urdf_collision = URDF.load(self._urdf_collision_path)
        self.urdf_visual = URDF.load(self._urdf_visual_path)
    
    def get_fixed_link_tf(self, link_name: str):
        """ Retrieves the fixed transformation of a specified link in the format [x, y, z, x, y, z, w], 
        where the first three elements are the position coordinates and the last four elements 
        represent the orientation as a quaternion.

        Args:
            link_name [str]: The name of the link for which the fixed transformation is to be obtained.

        Return:
            list: A list containing the position (x, y, z) and orientation (quaternion x, y, z, w) 
            of the specified link relative to the world frame.
        
        """
        T = self.urdf.get_transform(frame_to=link_name, frame_from="world")
        q, xyz = TransformationMatrix2QuaternionXYZ(T)
        return [xyz[0], xyz[1], xyz[2], q[1], q[2], q[3], q[0]]
    
    def get_link_in_scene(self, link_name: str) -> trimesh.Trimesh:
        """ Retrieves the trimesh object of a specified link in the scene frame. 
        This involves loading the mesh file associated with the link, 
        applying the appropriate scale and transformation to align it with the scene, 
        and returning a simplified trimesh object.

        Args:
            link_name [str]: The name of the link whose trimesh object is to be retrieved.

        Return:
            trimesh.Trimesh: The transformed and scaled trimesh object of the specified link in the scene frame.
        """
        link: Link = self.urdf.link_map[link_name]
        link_visuals = link.visuals[0]
        link_origin = link_visuals.origin
        link_geometry = link_visuals.geometry
        link_mesh = link_geometry.mesh
        link_filename = link_mesh.filename
        link_scale = link_mesh.scale
        link_file_path = self._resolve_link_mesh_path(link_filename)
        link_trimesh = trimesh.load(link_file_path, force="mesh")
        link_trimesh.apply_scale(link_scale)
        link_trimesh.apply_transform(
            self.urdf_visual.get_transform(
                frame_to=link_name, 
                frame_from="world"
            ) @ link_origin
        )
        return self._simple_trimesh(link_trimesh)
    
    @staticmethod
    def get_link_assert_name(urdf_file, link_name):
        """ This static method parses a URDF (Unified Robot Description Format) XML file to find a specific link by its name.
        It then retrieves the filename of the mesh associated with the visual geometry of that link, if available,
        and returns the base name (without path and extension) of the mesh file.

        Args:
            urdf_file [str]: The file path to the URDF XML file.
            link_name [str]: The name of the link to search for in the URDF file.

        Return:
            str or None: The base name of the mesh file associated with the specified link's visual geometry,
            or None if the link or mesh is not found.
        """
        tree = etree.parse(urdf_file)
        root = tree.getroot()

        for link in root.findall('link'):
            if link.get('name') == link_name:
                visual = link.find('visual')
                if visual is not None:
                    geometry = visual.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            file_path = mesh.get('filename')
                            return file_path.split('/')[-1].split('.')[0]
        return None
    
    def get_link(self, link_name: str) -> trimesh.Trimesh:
        """ Retrieves the `trimesh.Trimesh` object corresponding to the specified link name from the URDF model.
        The mesh is loaded, scaled, and transformed according to the link's visual properties.

        Args:
            link_name [str]: The name of the link whose mesh is to be retrieved.

        Return:
            trimesh.Trimesh: The processed mesh object of the specified link in the object frame.

        """
        link: Link = self.urdf.link_map[link_name]
        link_visuals = link.visuals[0]
        link_origin = link_visuals.origin
        link_geometry = link_visuals.geometry
        link_mesh = link_geometry.mesh
        link_filename = link_mesh.filename
        link_scale = link_mesh.scale
        link_file_path = self._resolve_link_mesh_path(link_filename)
        link_trimesh = trimesh.load(link_file_path, force="mesh")
        link_trimesh.apply_scale(link_scale)
        link_trimesh.apply_transform(link_origin)
        return self._simple_trimesh(link_trimesh)
    
    def _simple_trimesh(self, link_trimesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """ This function creates and returns a new simple trimesh object using the vertices, faces, 
        and vertex colors from the input trimesh. The new mesh is processed for consistency.

        Args:
            link_trimesh [trimesh.Trimesh]: The input trimesh object from which to extract geometry and color information.

        Return:
            trimesh.Trimesh: A new trimesh object with the same vertices, faces, and vertex colors as the input, 
            processed for consistency.
        """
        return trimesh.Trimesh(
            vertices=link_trimesh.vertices,
            faces=link_trimesh.faces,
            vertex_colors=link_trimesh.visual.to_color().vertex_colors,
            process=True,
        )
    
    def get_object_points_in_scene(
        self, 
        attach_link_name: str,
        sample_num: int=1024,
    ) -> np.ndarray:
        """ Samples a specified number of points uniformly from the surface of the given link's mesh in the scene.

        Args:
            attach_link_name [str]: The name of the link in the scene from which to sample points.
            sample_num [int]: The number of points to sample from the link's surface. Default is 1024.

        Return:
            np.ndarray: An array of sampled points (shape: [sample_num, 3]) from the surface of the specified link's mesh.
        """
        attach_link_trimesh = self.get_link_in_scene(attach_link_name)
        attach_link_points, _ = trimesh.sample.sample_surface(attach_link_trimesh, sample_num)
        return np.asarray(attach_link_points)
    
    def get_object_points(
        self, 
        attach_link_name: str,
        sample_num: int=1024,
    ) -> np.ndarray:
        """ Samples a specified number of points uniformly from the surface of the mesh associated with a given link name.

        Args:
            attach_link_name [str]: The name of the link whose mesh surface will be sampled.
            sample_num [int]: The number of points to sample from the mesh surface. Default is 1024.

        Return:
            np.ndarray: An array of sampled points from the surface of the specified link's mesh.
        """
        attach_link_trimesh = self.get_link(attach_link_name)
        attach_link_points, _ = trimesh.sample.sample_surface(attach_link_trimesh, sample_num)
        return np.asarray(attach_link_points)
    
    def crop_scene_and_sample_points(
        self,
        transformation_matrix: Union[List, np.ndarray],
        sample_num: int,
        LWH: List[float],
        min_z: float = -0.01,
        sample_color: bool = False, # when sample_color is true, there is wrong because of trimesh.
    ) -> np.ndarray:
        """ Crops the scene using a set of slicing planes defined in the agent's local view, 
        applies a transformation to these planes, and samples points from the resulting cropped mesh surface. 
        The sampled point clouds are returned in the world coordinate system. 
        Optionally, color information for the sampled points can also be returned.

        Args:
            transformation_matrix [Union[List, np.ndarray]]: The transformation matrix (4x4) to convert slicing planes from local to world coordinates.
            sample_num [int]: The number of points to sample from the cropped scene surface.
            LWH [List[float]]: The length, width, and height of the cropping box.
            min_z [float, optional]: The minimum z-value for cropping. Defaults to -0.01.
            sample_color [bool, optional]: Whether to sample color information along with points. Defaults to False.
            
        Return:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: 
                If sample_color is False, returns the sampled points as an (N, 3) array.
                If sample_color is True, returns a tuple (points, colors), where points is an (N, 3) array and colors is an (N, 3) or (N, 4) array of color values.
        """
        slice_plane_set = {
            "plane1": [
                [-1 + LWH[0], 0, 0],
                [-1, 0, 0],
            ],
            "plane2": [
                [-1, 0, 0],
                [1, 0, 0],
            ],    
            "plane3": [
                [0, LWH[1] / 2, 0],
                [0, -1, 0],
            ],    
            "plane4": [
                [0, - LWH[1] / 2, 0],
                [0, 1, 0],
            ],    
            "plane5": [
                [0, 0, LWH[2]],
                [0, 0, -1],
            ],    
            "plane6": [
                [0, 0, min_z],
                [0, 0, 1],
            ],
        }

        scene_slice = self.trimesh_collision
        count = len(slice_plane_set.keys())
        for i in range(count):
            slice_plane = slice_plane_set[f"plane{i+1}"]
            plane_origin = np.asarray(slice_plane[0])
            plane_normal = np.asarray(slice_plane[1])
            plane_origin_homog = np.concatenate((plane_origin, np.ones(1)), axis=0)
            plane_origin_transform = np.matmul(np.asarray(transformation_matrix), plane_origin_homog)[:3]
            plane_normal_transform = np.matmul(np.asarray(transformation_matrix)[:3,:3], plane_normal)

            scene_slice = scene_slice.slice_plane(
                plane_origin_transform,
                plane_normal_transform,
            )
        
        if sample_color:    
            crop_scene_points, _, crop_scene_colors = trimesh.sample.sample_surface(
                scene_slice, sample_num, sample_color=sample_color
            )
            return crop_scene_points, crop_scene_colors
        else:
            crop_scene_points, _ = trimesh.sample.sample_surface(scene_slice, sample_num, sample_color=sample_color)
            return crop_scene_points

    def get_boundaries(self):
        """ Calculates the minimum and maximum boundary coordinates of the mesh surface by sampling points from the mesh.

        Args:
            None

        Return:
            min_boundaries [np.ndarray]: The minimum x, y, z coordinates among the sampled points.
            max_boundaries [np.ndarray]: The maximum x, y, z coordinates among the sampled points.
        """
        self._pcd, _ = trimesh.sample.sample_surface(self.trimesh_visual, 512)
        min_boundaries = np.min(np.asarray(self._pcd), axis=0)
        max_boundaries = np.max(np.asarray(self._pcd), axis=0)
        return min_boundaries, max_boundaries
    
    def sample_pointcloud_from_trimesh_visual(self, sample_num: int, sample_color: bool=False):
        """ Samples points from the surface of the mesh stored in self.trimesh_visual. 
        Optionally, it can also sample the color information of the points if requested.

        Args:
            sample_num [int]: The number of points to sample from the mesh surface.
            sample_color [bool]: Whether to sample color information for each point. If True, returns both points and their colors.

        Return:
            If sample_color is True, returns a tuple (points, colors), where points is an array of sampled point coordinates and 
            colors is an array of corresponding color values.
            If sample_color is False, returns only the array of sampled point coordinates.
        """
        if sample_color:    
            points, _, colors = trimesh.sample.sample_surface(
                self.trimesh_visual, sample_num, sample_color=sample_color
            )
            return points, colors
        else:
            points, _ = trimesh.sample.sample_surface(
                self.trimesh_visual, sample_num, sample_color=sample_color
            )
            return points
    
    def _del_link_from_urdf(self, key:str, origin_path: str, save_path: str, isoverwrite: bool=False):
        """ Removes a link and its associated fixed joint from a URDF XML file if the joint's parent is "world" 
        and the child link name contains the specified key and "link". The modified URDF is saved to the specified path.

        Args:
            key [str]: The key to identify the link and joint to be removed.
            origin_path [str]: The file path to the original URDF file.
            save_path [str]: The file path where the modified URDF will be saved.
            isoverwrite [bool, optional]: Whether to overwrite the save_path file if it exists. Defaults to False.

        Return:
            None: The function writes the modified URDF to the save_path and does not return a value.
        """
        if (not os.path.isfile(save_path)) or (isoverwrite):
            main_urdf_path = origin_path
            if not os.path.isfile(main_urdf_path):
                raise FileNotFoundError(
                    f"Cannot build '{save_path}' because source URDF does not exist: '{main_urdf_path}'"
                )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            main_urdf = copy.deepcopy(etree.parse(source=main_urdf_path))
            root_main = main_urdf.getroot()
            floor_joints = main_urdf.xpath(f"//joint[contains(@name,{key}) and @type='fixed']")
            
            for floor_joint in floor_joints:            
                if floor_joint.find("parent").attrib["link"] == "world":
                    clild_name = floor_joint.find("child").attrib["link"]
                    if key in clild_name and "link" in clild_name:                   
                        frmat_link = "//link[@name='{}']".format(clild_name)                    
                        floor_link = main_urdf.xpath(frmat_link)[0]                    
                        root_main.remove(floor_link)
                        root_main.remove(floor_joint)        
            main_urdf.write(save_path, pretty_print=True, xml_declaration=True, encoding='utf-8')

    def _create_urdf_without_flooring(self):
        """ Removes the 'room' link from the URDF file and its corresponding collision file, effectively creating a URDF without the flooring.
        """
        self._del_link_from_urdf('room', self._urdf_path, self._urdf_collision_path)
    
    def _create_urdf_without_ceiling(self):
        """ Removes the 'ceiling' link from the URDF files associated with the scene, specifically from the collision and visual URDF paths.
        """
        self._del_link_from_urdf('ceiling', self._urdf_collision_path, self._urdf_visual_path)
        # self._del_link_from_urdf('ceiling', self._urdf_path, self._urdf_visual_path, True)
