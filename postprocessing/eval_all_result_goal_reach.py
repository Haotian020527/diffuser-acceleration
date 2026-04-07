import csv
import itertools
import sys
import json
import os
import argparse
import random
import sys
import numpy as np
import torch
from tqdm import tqdm
from typing import Any, Dict, Sequence
from math import pi
sys.path.append("../")
from utils.io import dict2json
from utils.transform import SE3
from env.sampler.mk_sampler import MecKinovaSampler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--robot', type=str, default='MecKinova', help='robot name, such as MecKinova or Franka')
    p.add_argument('--task', type=str, default='goal-reach', help='task name, such as pick, place or goal-reach')
    p.add_argument('--result_dir', type=str, help='inference result directory path')
    p.add_argument('--dataset_test_dir', type=str, help='dataset test directory')

    opt = p.parse_args()
    return opt

def percent_true(arr: Sequence) -> float:
    return 100 * np.count_nonzero(arr) / len(arr)

def add_metric(group, key, value):
    group[key] = group.get(key, []) + [value]

def eval_metrics(group: Dict[str, Any]) -> Dict[str, float]:
    # There was a problem with the previous code, so let's rework it here
    group["physical_violations"] = (
        group["collision"] 
        or group["joint_limit_violation"]
        or group["self_collision"]
    )
    group["physical_success"] = [not x for x in group["physical_violations"]]
    ## --------------------------------------------------------------------

    physical_success = percent_true(group["physical_success"])
    physical = percent_true(group["physical_violations"])
    config_smoothness = np.mean(group["config_smoothness"])
    eff_smoothness = np.mean(group["eff_smoothness"])
    all_eff_position_path_lengths = np.asarray(group["eff_position_path_length"])
    all_eff_orientation_path_lengths = np.asarray(
        group["eff_orientation_path_length"]
    )
    all_times = np.asarray(group["time"])

    is_smooth = percent_true(
        np.logical_and(
            np.asarray(group["config_smoothness"]) < -1.6,
            np.asarray(group["eff_smoothness"]) < -1.6,
        )
    )

    physical_successes = group["physical_success"]

    physical_success_position_path_lengths = all_eff_position_path_lengths[
        list(physical_successes)
    ]
    physical_success_orientation_path_lengths = all_eff_orientation_path_lengths[
        list(physical_successes)
    ]
    
    if len(physical_success_position_path_lengths) > 0:
        eff_position_path_length = (
            np.mean(physical_success_position_path_lengths),
            np.std(physical_success_position_path_lengths),
        )
    else:
        eff_position_path_length = (0, 0)

    task_success = np.asarray(group["task_success"])

    is_success = percent_true(
        np.logical_and(physical_successes, task_success)
    )

    if len(physical_success_orientation_path_lengths) > 0:
        eff_orientation_path_length = (
            np.mean(physical_success_orientation_path_lengths),
            np.std(physical_success_orientation_path_lengths),
        )
    else:
        eff_orientation_path_length = (0, 0)    

    physical_success_times = all_times[list(group["physical_success"])]

    if len(physical_success_times) > 0:
        time = (
            np.mean(physical_success_times),
            np.std(physical_success_times),
        )
    else:    
        time = (0, 0)

    collision = percent_true(group["collision"])
    joint_limit = percent_true(group["joint_limit_violation"])
    self_collision = percent_true(group["self_collision"])
    depths = np.array(
        list(itertools.chain.from_iterable(group["collision_depths"]))
    )
    if len(depths) == 0: depths = [0]
    all_num_steps = np.asarray(group["num_steps"])
    physical_success_num_steps = all_num_steps[list(physical_successes)]
    if len(physical_success_num_steps) > 0:
        step_time = (
            np.mean(physical_success_times / physical_success_num_steps),
            np.std(physical_success_times / physical_success_num_steps),
        )
    else:
        step_time = (0, 0)

    return {
        "time": time,
        "step time": step_time,
        "is success": is_success,
        "physical_success": physical_success,
        "env collision": collision,
        "self collision": self_collision,
        "joint violation": joint_limit,
        "physical violations": physical,
        "average collision depth": 100 * np.mean(depths),
        "median collision depth": 100 * np.median(depths),
        "is smooth": is_smooth,
        "average config sparc": config_smoothness,
        "average eff sparc": eff_smoothness,
        "eff position path length": eff_position_path_length,
        "eff orientation path length": eff_orientation_path_length,
    }

def print_metrics(group: Dict[str, Any]):
    metrics = eval_metrics(group)
    print(f"% Success: {metrics['is success']:4.2f}")
    print(f"% With Environment Collision: {metrics['env collision']:4.2f}")
    print(f"% With Self Collision: {metrics['self collision']:4.2f}")
    print(f"% With Joint Limit Violations: {metrics['joint violation']:4.2f}")
    print(f"Average Collision Depth (cm): {metrics['average collision depth']}")
    print(f"Median Collision Depth (cm): {metrics['median collision depth']}")
    print(f"Average Config SPARC: {metrics['average config sparc']:4.2f}")
    print(f"Average End Eff SPARC: {metrics['average eff sparc']:4.2f}")
    print(f"% Smooth: {metrics['is smooth']:4.2f}")
    print(
        "Average End Eff Position Path Length:"
        f" {metrics['eff position path length'][0]:4.2f}"
        f" ± {metrics['eff position path length'][1]:4.2f}"
    )
    print(
        "Average End Eff Orientation Path Length:"
        f" {metrics['eff orientation path length'][0]:4.2f}"
        f" ± {metrics['eff orientation path length'][1]:4.2f}"
    )
    print(f"Average Time: {metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}")
    print(
        "Average Time Per Step (Not Always Valuable):"
        f" {metrics['step time'][0]:4.6f}"
        f" ± {metrics['step time'][1]:4.6f}"
    )

def get_metrics(group: Dict[str, Any]):
    metrics = eval_metrics(group)
    return {
        "% Success": f"{metrics['is success']:4.2f}",
        "% With Environment Collision": f"{metrics['env collision']:4.2f}",
        "% With Self Collision": f"{metrics['self collision']:4.2f}",
        "% With Joint Limit Violations": f"{metrics['joint violation']:4.2f}",
        "Average Collision Depth (cm)": f"{metrics['average collision depth']}",
        "Median Collision Depth (cm)": f"{metrics['median collision depth']}",
        "Average Config SPARC": f"{metrics['average config sparc']:4.2f}",
        "Average End Eff SPARC": f"{metrics['average eff sparc']:4.2f}",
        "% Smooth": f"{metrics['is smooth']:4.2f}",
        "Average End Eff Position Path Length": f"{metrics['eff position path length'][0]:4.2f} ± {metrics['eff position path length'][1]:4.2f}",
        "Average End Eff Orientation Path Length": f" {metrics['eff orientation path length'][0]:4.2f} ± {metrics['eff orientation path length'][1]:4.2f}",
        "Average Time": f"{metrics['time'][0]:4.2f} ± {metrics['time'][1]:4.2f}",
        "Average Time Per Step (Not Always Valuable)": f" {metrics['step time'][0]:4.6f} ± {metrics['step time'][1]:4.6f}"
    }

def save_metrics(save_path: str, eval_group: Dict[str, Any]):
    item = {}
    for key in eval_group.keys():
        item[key] = get_metrics(eval_group[key])
    dict2json(save_path, item)

def eval_pick_result(result_dir: str, test_dir: str, sampler: MecKinovaSampler):
    object_result_dir = os.path.join(result_dir, 'object')

    eval_group = {}; eval_group["all"] = {}
    id_iter = tqdm(sorted(os.listdir(object_result_dir), key=lambda x:int(x.split('.')[0])), desc="{0: ^10}".format('ID'), leave=False, mininterval=0.1)

    target_error = []
    task_success_list = []
    for id_name in id_iter:
        id = id_name.split('.')[0]
        result_file_path = os.path.join(object_result_dir, id_name)
        with open(os.path.join(result_file_path), "r") as f:
            result_item = json.load(f)
        
        test_file_path = os.path.join(test_dir, id + '.npy')
        test_data = np.load(test_file_path, allow_pickle=True).item()
        
        cur_cfg = torch.as_tensor(result_item['trajectory_w'][-1], device='cuda:0')
        goal_cfg = torch.as_tensor(test_data['trajectory']['traj_w'][-1], device='cuda:0')
        t_dist_p, r_dist_p = compute_distance_positive(sampler, cur_cfg, goal_cfg)
        t_dist_n, r_dist_n = compute_distance_negative(sampler, cur_cfg, goal_cfg)

        t_error = 0.04
        r_error = 20
        if (t_dist_p < t_error and r_dist_p < r_error):
            task_success = 1
        elif (t_dist_n < t_error and r_dist_n < r_error):
            task_success = 1
        else:
            task_success = 0

        if r_dist_p < r_dist_n:
            target_error.append([t_dist_p, r_dist_p, 'model'])
        else: 
            target_error.append([t_dist_n, r_dist_n, 'model'])
        
        task_success_list.append(task_success)

        # all results
        add_metric(eval_group["all"], "collision_depths", result_item["collision_depths"])
        add_metric(eval_group["all"], "collision", result_item["collision"])
        add_metric(eval_group["all"], "physical_success", result_item["physical_success"])
        add_metric(eval_group["all"], "physical_violations", result_item["physical_violations"])
        add_metric(eval_group["all"], "joint_limit_violation", result_item["joint_limit_violation"])
        add_metric(eval_group["all"], "self_collision", result_item["self_collision"])
        add_metric(eval_group["all"], "config_smoothness", result_item["config_smoothness"])
        add_metric(eval_group["all"], "eff_smoothness", result_item["eff_smoothness"])
        add_metric(eval_group["all"], "joint_limit_violation", result_item["joint_limit_violation"])
        add_metric(eval_group["all"], "eff_position_path_length", result_item["eff_position_path_length"])
        add_metric(eval_group["all"], "eff_orientation_path_length", result_item["eff_orientation_path_length"])
        add_metric(eval_group["all"], "time", result_item["time"])
        add_metric(eval_group["all"], "num_steps", result_item["num_steps"])
        add_metric(eval_group["all"], "task_success", task_success)

    save_metrics(os.path.join(result_dir, 'eval_metrics_v2.json'), eval_group)
    dict2json(os.path.join(result_dir, 'task_success.json'), task_success_list)

    with open(os.path.join(result_dir, 'target_error_v2.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(target_error)

def compute_distance_positive(sampler: MecKinovaSampler, cur_cfg: torch.Tensor, goal_cfg: torch.Tensor):
    cur_pose = sampler.end_effector_pose(cur_cfg).squeeze(0).clone().detach().cpu().numpy()
    goal_pose = sampler.end_effector_pose(goal_cfg).squeeze(0).clone().detach().cpu().numpy()

    cur_se3 = SE3(matrix=cur_pose)
    goal_se3 = SE3(matrix=goal_pose)

    t_dist = np.linalg.norm(cur_se3._xyz - goal_se3._xyz)
    r_dist = np.abs(np.degrees((cur_se3.so3._quat * goal_se3.so3._quat.conjugate).radians))

    return (
        t_dist, 
        r_dist
    )

def compute_distance_negative(sampler: MecKinovaSampler, cur_cfg: torch.Tensor, goal_cfg: torch.Tensor):
    cur_cfg[-1] += pi
    if cur_cfg[-1] > pi:
        cur_cfg[-1] -= 2 * pi
    cur_pose = sampler.end_effector_pose(cur_cfg).squeeze(0).clone().detach().cpu().numpy()
    goal_pose = sampler.end_effector_pose(goal_cfg).squeeze(0).clone().detach().cpu().numpy()

    cur_se3 = SE3(matrix=cur_pose)
    goal_se3 = SE3(matrix=goal_pose)

    t_dist = np.linalg.norm(cur_se3._xyz - goal_se3._xyz)
    r_dist = np.abs(np.degrees((cur_se3.so3._quat * goal_se3.so3._quat.conjugate).radians))

    return (
        t_dist, 
        r_dist
    )


if __name__ == "__main__":
    args = parse_args()

    ## set random seed
    seed = 2024
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    mk_sampler = MecKinovaSampler('cuda', num_fixed_points=1024, use_cache=True)
    eval_pick_result(args.result_dir, args.dataset_test_dir, mk_sampler)
