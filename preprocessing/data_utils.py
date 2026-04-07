import os
import time
import sys
import numpy as np
import mesh2sdf
from typing import Dict, List, Optional
from env.scene.base_scene import Scene
from utils.path import RootPath
sys.path.append("..")

def compute_data_number(data_path:str, task_name:str) -> int:
    """ Compute the number of data.
    """
    data_num = 0
    scene_names = os.listdir(data_path)

    for scene_name in scene_names:
        scene_dir_path = os.path.join(data_path, scene_name)
        obj_names = os.listdir(scene_dir_path)
        for obj_name in obj_names:
            obj_dir_path = os.path.join(scene_dir_path, obj_name)
            timestamp_names = os.listdir(obj_dir_path)
            for timestamp_name in timestamp_names:
                timestamp_dir_path = os.path.join(obj_dir_path, timestamp_name)
                id_names = os.listdir(timestamp_dir_path)
                for id_name in id_names:
                    data_dir_path = os.path.join(timestamp_dir_path, id_name)
                    if os.path.isdir(data_dir_path):
                        if check_file("trajectory", data_dir_path) \
                        and check_file("config.json", data_dir_path) \
                        and check_file(task_name + "_vkc_return.json", data_dir_path) \
                        and check_file("vkc_request.json", data_dir_path):
                            data_num += 1
    return data_num

def compute_data_number_in_scene_object_list(
    data_path:str, 
    task_name:str, 
    scene_list:List[str],
    object_list:List[str],
) -> int:
    """ Compute the number of data.
    """
    num = 0 # number of trajectories in seen scenes and objects
    scene_names = os.listdir(data_path)
    for scene_name in scene_names:
        scene_dir_path = os.path.join(data_path, scene_name)
        obj_names = os.listdir(scene_dir_path)
        for obj_name in obj_names:
            obj_assert_name = Scene.get_link_assert_name(str(RootPath.SCENE / scene_name / "main.urdf"), obj_name)
            obj_dir_path = os.path.join(scene_dir_path, obj_name)
            timestamp_names = os.listdir(obj_dir_path)
            for timestamp_name in timestamp_names:
                timestamp_dir_path = os.path.join(obj_dir_path, timestamp_name)
                id_names = os.listdir(timestamp_dir_path)
                for id_name in id_names:
                    data_dir_path = os.path.join(timestamp_dir_path, id_name)
                    if os.path.isdir(data_dir_path):
                        if check_file("trajectory", data_dir_path) \
                        and check_file("config.json", data_dir_path) \
                        and check_file(task_name + "_vkc_return.json", data_dir_path) \
                        and check_file("vkc_request.json", data_dir_path):
                            if (scene_name in scene_list) and (obj_assert_name in object_list):
                                num += 1
    return num

def check_file(name:str, path:Optional[str]) -> bool:
    """ Check whether the file or directory in path exists.

    Args:
        name [str]: Name of the file or directory that you want to check.
        path [Optinal[str]]: Path to be checked. If path is None, the current path is used by default.

    Returns:
        bool. Returns true if the path to be checked contains the destination file or directory. False is 
            returned when the path to be checked does not contain the destination file or directory.      
    """
    if path is None:
        path = os.getcwd()
    if os.path.exists(path + '/' + name):
        # print("Under the path: " + path + '\n' + name + " is exist")
        return True
    else:
        if (os.path.exists(path)):
            print("Under the path: " + path + '\n' + name + " is not exist")
        else:
            print("This path could not be found: " + path + '\n')
        return False

def compute_scene_sdf(scene: Scene, size: int=128) -> Dict:
    """ Compute scene SDF value for collision loss computation.
    """
    # NOTE: The floor cannot be considered when calculating SDF, 
    # because the floor and the agent must touch.
    mesh = scene.trimesh_collision
    mesh_scale = 0.8
    level = 2 / size

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    # fix mesh
    start_time = time.time()
    sdf, mesh = mesh2sdf.compute(vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)
    end_time = time.time()
    # output
    mesh.vertices = mesh.vertices / scale + center

    sdf_dict = {
        'sdf_norm_value': sdf,
        'scene_mesh_center': np.array(center),
        'scene_mesh_scale': np.array(scale),
        'resolution': np.array(size),
    }
    print('It takes %.4f seconds to process %s' % (end_time - start_time, scene.name))
    return sdf_dict