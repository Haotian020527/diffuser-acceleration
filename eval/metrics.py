import time
import pickle
import itertools
import pybullet as p
import numpy as np
from cprint import *
from pathlib import Path
from eval.sparc import sparc
from utils.transform import SE3
from termcolor import colored
from geometrout.primitive import Sphere
from typing import Sequence, Union, List, Tuple, Any, Dict, Optional
from env.agent.mec_kinova import MecKinova
from env.sim.bullet_simulator import Bullet


def percent_true(arr: Sequence) -> float:
    """ Calculates the percentage of True values in a boolean sequence or the percentage of nonzero 
    elements in a numerical sequence.

    Args:
        arr [Sequence]: The input sequence, which can be a sequence of booleans or numerics.

    Return:
        float: The percentage of True or nonzero elements in the input sequence.
    """
    return 100 * np.count_nonzero(arr) / len(arr)


class Evaluator:
    """ This class can be used to evaluate a whole set of environments and data.
    """

    def __init__(self, gui: bool = False):
        """ Initializes the evaluator class.

        Args:
            gui [bool]: Whether to enable GUI visualization for trajectories and visually indicate failures.

        Return:
            None. Initializes the evaluator class and sets up simulation environments and robots for evaluation.
        """
        self.sim = Bullet(gui=False)
        self.sim_robot = self.sim.load_robot(MecKinova)
        self.robot = MecKinova()

        self.self_collision_sim = Bullet(gui=False)
        self.self_collision_robot = self.self_collision_sim.load_robot(MecKinova)
        
        self.groups = {}
        self.current_result = {}
        self.current_group = None
        self.current_group_key = None

        if gui:
            self.gui_sim = Bullet(gui=True)
            p.configureDebugVisualizer(
                flag=p.COV_ENABLE_WIREFRAME, 
                enable=1, 
                physicsClientId=self.gui_sim.clid
            )
            self.gui_robot = self.gui_sim.load_robot(MecKinova)
        else:
            self.gui_sim = None
            self.gui_robot = None

    def run_visuals(
        self,
        dt: float,
        trajectory: Sequence[Union[Sequence, np.ndarray]],
        obstacles_path: str,
        obstacles_pose:Optional[SE3]=None,
    ):
        """ Visualizes a robot trajectory in a GUI, displaying obstacles and highlighting collision points.
        The color of the visualization changes based on whether a collision occurs during the trajectory.

        Args:
            dt [float]: The approximate timestep to use when visualizing the trajectory.
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory to visualize, as a sequence of joint configurations.
            obstacles_path [str]: The file path to the obstacle URDF to be loaded in the simulation.
            obstacles_pose [Optional[SE3]]: The pose of the obstacles in the simulation environment.

        Return:
            None. The function visualizes the trajectory and obstacles in the GUI.
        """
        self.gui_sim.clear_all_obstacles()
        collision_ids = self.gui_sim.load_urdf_obstacle(obstacles_path, obstacles_pose)

        for i, q in enumerate(trajectory):
            self.gui_robot.marionette(q)
            collision_points = self.gui_robot.get_collision_points(
                collision_ids, check_self=False
            )
            if len(collision_points):
                collision_points = np.array(collision_points, dtype=np.double)
                self.gui_sim.load_primitives(
                    [Sphere(center=c, radius=0.02) for c in collision_points],
                    [1, 0, 0, 0.5],
                )
                time.sleep(0.25)
            time.sleep(dt)

    def create_new_group(self, key: str):
        """ Creates a new metric group with the specified key, initializing an empty dictionary for the group,
        and sets it as the current active group.

        Args:
            key [str]: The key used to identify and create the new metric group.

        Return:
            None. The function updates the internal state of the object by adding a new group and setting it as current.
        """
        self.groups[key] = {}
        self.current_group_key = key
        self.current_group = self.groups[key]

    def in_collision(self, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> bool:
        """ Checks whether any configuration in the given trajectory results in a collision,
        using all included collision checkers (logical AND across methods). Iterates through
        each configuration, applies it to the simulated robot, and checks for collisions.

        Args:
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory to be checked, 
                where each element is a robot configuration.

        Return:
            bool: True if any configuration in the trajectory is in collision, otherwise False.
        """
        for i, q in enumerate(trajectory):
            self.sim_robot.marionette(q)
            if self.sim.in_collision(
                self.sim_robot, check_self=True
            ):
                return True
        return False

    def has_self_collision(self, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> bool:
        """ Checks whether any configuration in the given trajectory results in a self-collision for the robot.
        Iterates through each configuration, applies it to the robot model, and checks for self-collision.

        Args:
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: 
                The trajectory to be checked, where each element is a robot configuration.

        Return:
            bool: True if any configuration in the trajectory results in a self-collision, otherwise False.
        """
        for i, q in enumerate(trajectory):
            self.self_collision_robot.marionette(q)
            if self.self_collision_sim.in_collision(
                self.self_collision_robot, check_self=True
            ):
                return True
        return False

    def get_collision_depths(self, trajectory: Sequence[Union[Sequence, np.ndarray]]) -> List[float]:
        """ This function computes and returns all collision depths encountered along a given trajectory.
        It iterates through each configuration in the trajectory, sets the robot to that configuration,
        and collects the collision depths with respect to the specified obstacles.

        Args:
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory, represented as a sequence 
                of robot configurations.

        Return:
            List[float]: A list containing all collision depths detected along the trajectory.
            
        """
        all_depths = []
        for i, q in enumerate(trajectory):
            self.sim_robot.marionette(q)
            depths = self.sim_robot.get_collision_depths(self.sim.obstacle_ids)
            all_depths.extend(depths)
        return all_depths

    @staticmethod
    def violates_joint_limits(trajectory: Sequence[Union[Sequence, np.ndarray]]) -> bool:
        """ Checks whether any configuration in the given trajectory violates the joint limits.

        Args:
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory to be checked, 
                where each element represents a joint configuration.

        Return:
            bool: Returns True if any configuration in the trajectory violates the joint limits, otherwise returns False.
        """
        for i, q in enumerate(trajectory):
            if not MecKinova.within_limits(q):
                return True
        return False

    def has_physical_violation(
        self, trajectory: Sequence[Union[Sequence, np.ndarray]], 
    ) -> bool:
        """ Checks whether the given trajectory has any physical violations, including collision with obstacles, 
        self-collision, or violation of joint limits.

        Args:
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory to be checked for physical violations.

        Return:
            bool: Returns True if there is at least one physical violation in the trajectory; otherwise, returns False.
        """
        return (
            self.in_collision(trajectory[:-1,:])
            or self.violates_joint_limits(trajectory)
            or self.has_self_collision(trajectory)
        )

    @staticmethod
    def calculate_smoothness(
        agent_object: object,
        trajectory: Sequence[Union[Sequence, np.ndarray]], 
        dt: float
    ) -> Tuple[float, float]:
        """ This function calculates the smoothness of a given trajectory using the SPARC (Spectral Arc Length) metric. 
        It computes smoothness in both the configuration space and the end-effector space for a MecKinova agent.

        Args:
            agent_object [object]: The agent object, which must be of type MecKinova.
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory to be evaluated, represented as a sequence of configurations.
            dt [float]: The time interval between consecutive steps in the trajectory.

        Return:
            Tuple[float, float]: The SPARC smoothness values in configuration space and end-effector space, respectively.
        """
        assert agent_object is MecKinova, "Agent type is invalid"
        mec_kinova = MecKinova()
        configs = np.asarray(trajectory)
        assert configs.ndim == 2 and configs.shape[1] == MecKinova.DOF
        config_movement = np.linalg.norm(np.diff(configs, 1, axis=0) / dt, axis=1)
        assert len(config_movement) == len(configs) - 1
        config_sparc, _, _ = sparc(config_movement, 1.0 / dt)

        eff_positions = np.asarray(
            [mec_kinova.get_eff_pose(q)[:3, 3] for q in trajectory]
        )
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        eff_movement = np.linalg.norm(np.diff(eff_positions, 1, axis=0) / dt, axis=1)
        assert len(eff_movement) == len(eff_positions) - 1
        eff_sparc, _, _ = sparc(eff_movement, 1.0 / dt)

        return config_sparc, eff_sparc

    def calculate_eff_path_lengths(
        self, agent_object: object, 
        trajectory: Sequence[Union[Sequence, np.ndarray]]
    ) -> Tuple[float, float]:
        """ Calculate the end effector path lengths for both position and orientation along a given trajectory.

        Args:
            agent_object [object]: The agent object, must be of type MecKinova.
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The sequence of joint configurations representing the trajectory.

        Return:
            Tuple[float, float]: A tuple containing the total path length of the end effector's position (in the workspace) and 
            the total orientation change (in degrees) along the trajectory.
        """
        assert agent_object is MecKinova, "Agent type is invalid"
        eff_poses = [SE3(self.robot.get_eff_pose(q)) for q in trajectory]
        eff_positions = np.asarray([pose._xyz for pose in eff_poses])
        assert eff_positions.ndim == 2 and eff_positions.shape[1] == 3
        position_step_lengths = np.linalg.norm(
            np.diff(eff_positions, 1, axis=0), axis=1
        )
        eff_position_path_length = sum(position_step_lengths)

        eff_quaternions = [pose.so3._quat for pose in eff_poses]
        eff_orientation_path_length = 0
        for qi, qj in zip(eff_quaternions[:-1], eff_quaternions[1:]):
            eff_orientation_path_length += np.abs(
                np.degrees((qj * qi.conjugate).radians)
            )
        return eff_position_path_length, eff_orientation_path_length

    def evaluate_trajectory(
        self,
        dt: float,
        time: float,
        trajectory: Sequence[Union[Sequence, np.ndarray]],
        agent_object: object,
        obstacles_path: str,
        obstacles_pose: Optional[SE3]=None,
        skip_metrics: bool = False,
    ):
        """ Evaluates a single trajectory for a robotic agent, computes various path and physical metrics, 
        and stores the results in the current evaluation group. Optionally visualizes and prints 
        relevant information if GUI is enabled.

        Args:
            dt [float]: The time step for the trajectory.
            time [float]: The time taken to calculate the trajectory.
            trajectory [Sequence[Union[Sequence, np.ndarray]]]: The trajectory to be evaluated.
            agent_object [object]: The agent (robot) object for which the trajectory is evaluated.
            obstacles_path [str]: The file path to the obstacles' URDF description.
            obstacles_pose [Optional[SE3]]: The pose of the obstacles in the scene.
            skip_metrics [bool]: Whether to skip the path metrics (e.g., if the planner failed).

        Return:
            dict: The evaluation result for the current trajectory, containing computed metrics.
        """
        # assert agent_object is MecKinova, "Agent type is invalid"
        def add_metric(key, value):
            self.current_group[key] = self.current_group.get(key, []) + [value]
            self.current_result[key] = value

        if skip_metrics:
            add_metric("physical_success", False)
            add_metric("time", np.inf)
            add_metric("skips", True)
            return
        self.sim.clear_all_obstacles()
        self.sim.load_urdf_obstacle(path=obstacles_path, pose=obstacles_pose)

        # the last frame of picking and placing motion needs collision
        in_collision = self.in_collision(trajectory[:-1,:])
        if in_collision:
            collision_depths = (
                self.get_collision_depths(trajectory[:-1,:]) if in_collision else []
            )
        else:
            collision_depths = []
        add_metric("collision_depths", collision_depths)
        add_metric("collision", in_collision)
        add_metric("joint_limit_violation", self.violates_joint_limits(trajectory))
        add_metric("self_collision", self.has_self_collision(trajectory))

        physical_violation = self.has_physical_violation(trajectory)
        add_metric("physical_violations", physical_violation)

        config_smoothness, eff_smoothness = self.calculate_smoothness(agent_object, trajectory, dt)
        add_metric("config_smoothness", config_smoothness)
        add_metric("eff_smoothness", eff_smoothness)

        (
            eff_position_path_length,
            eff_orientation_path_length,
        ) = self.calculate_eff_path_lengths(agent_object, trajectory)
        add_metric("eff_position_path_length", eff_position_path_length)
        add_metric("eff_orientation_path_length", eff_orientation_path_length)

        physical_success = not physical_violation

        add_metric("physical_success", physical_success)
        add_metric("time", time)
        add_metric("num_steps", len(trajectory))

        def msg(metric, is_error):
            return colored(metric, "red") if is_error else colored(metric, "green")

        if self.gui_sim is not None and len(collision_depths) > 0:
            print("Config SPARC:", msg(config_smoothness, config_smoothness > -1.6))
            print("End Eff SPARC:", msg(eff_smoothness, eff_smoothness > -1.6))
            print(f"End Eff Position Path Length: {eff_position_path_length}")
            print(f"End Eff Orientation Path Length: {eff_orientation_path_length}")
            print("Physical violation:", msg(physical_violation, physical_violation))
            print(
                "Collisions:",
                msg(
                    self.in_collision(trajectory),
                    self.in_collision(trajectory),
                ),
            )
            if len(collision_depths) > 0:
                print(f"Mean Collision Depths: {100 * np.mean(collision_depths)}")
                print(f"Median Collision Depths: {100 * np.median(collision_depths)}")
            print(
                "Joint Limit Violation:",
                msg(
                    self.violates_joint_limits(trajectory),
                    self.violates_joint_limits(trajectory),
                ),
            )
            print(
                "Self collision:",
                msg(
                    self.has_self_collision(trajectory),
                    self.has_self_collision(trajectory),
                ),
            )
            self.run_visuals(dt, trajectory, obstacles_path, obstacles_pose)
        
        return self.current_result # return current trajectory evaluation result


    @staticmethod
    def metrics(group: Dict[str, Any]) -> Dict[str, float]:
        """ Calculates various evaluation metrics for a specific group of results, including success rates, 
        smoothness, path lengths, collision statistics, and timing information. The function processes the 
        input group dictionary, computes statistics, and returns a dictionary of metric names mapped to their computed values.

        Args:
            group [Dict[str, Any]]: A dictionary containing lists or arrays of evaluation results for a group, including keys such as 
                "physical_success", "physical_violations", "config_smoothness", "eff_smoothness", 
                "eff_position_path_length", "eff_orientation_path_length", "time", "collision", 
                "joint_limit_violation", "self_collision", "collision_depths", "num_steps", and optionally "skips".

        Return:
            Dict[str, float]: A dictionary mapping metric names to their computed float values or tuples (mean, std) for metrics such 
                as time, step time, path lengths, and collision depths.
        """
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

        # Only use filtered physical_successes for eff path length because we only
        # save the metric when it's an unskipped trajectory
        # TODO maybe clean this up so that it's not so if/else-y and either
        # all metrics are saved or only the ones that aren't skipped save
        skips = []
        if "skips" in group:
            # Needed when there are skips
            physical_successes = np.asarray(group["physical_success"])
            unskipped_physical_successes = physical_successes[~np.isinf(all_times)]
            skips = group["skips"]
        else:
            unskipped_physical_successes = group["physical_success"]

        physical_success_position_path_lengths = all_eff_position_path_lengths[
            list(unskipped_physical_successes)
        ]

        physical_success_orientation_path_lengths = all_eff_orientation_path_lengths[
            list(unskipped_physical_successes)
        ]
        eff_position_path_length = (
            np.mean(physical_success_position_path_lengths),
            np.std(physical_success_position_path_lengths),
        )

        eff_orientation_path_length = (
            np.mean(physical_success_orientation_path_lengths),
            np.std(physical_success_orientation_path_lengths),
        )
        physical_success_times = all_times[list(group["physical_success"])]
        time = (
            np.mean(physical_success_times),
            np.std(physical_success_times),
        )

        collision = percent_true(group["collision"])
        joint_limit = percent_true(group["joint_limit_violation"])
        self_collision = percent_true(group["self_collision"])
        depths = np.array(
            list(itertools.chain.from_iterable(group["collision_depths"]))
        )
        all_num_steps = np.asarray(group["num_steps"])
        physical_success_num_steps = all_num_steps[list(unskipped_physical_successes)]
        step_time = (
            np.mean(physical_success_times / physical_success_num_steps),
            np.std(physical_success_times / physical_success_num_steps),
        )
        return {
            "physical_success": physical_success,
            "total": len(group["physical_success"]),
            "skips": len(skips),
            "time": time,
            "step time": step_time,
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

    @staticmethod
    def print_metrics(group: Dict[str, Any]):
        """ Print the evaluation metrics in a human-readable format.

        Args:
            group [Dict[str, Any]]: A dictionary containing the group of evaluation results.

        Return:
            None. The function prints the formatted metrics to the standard output.
        """
        metrics = Evaluator.metrics(group)
        print(f"Total problems: {metrics['total']}")
        print(f"# Skips (Hard Failures): {metrics['skips']}")
        print(f"% Physical_success: {metrics['physical_success']:4.2f}")
        print(f"% With Environment Collision: {metrics['env collision']:4.2f}")
        print(f"% With Self Collision: {metrics['self collision']:4.2f}")
        print(f"% With Joint Limit Violations: {metrics['joint violation']:4.2f}")
        print(f"Average Collision Depth (cm): {metrics['average collision depth']}")
        print(f"Median Collision Depth (cm): {metrics['median collision depth']}")
        print(f"% With Physical Violations: {metrics['physical violations']:4.2f}")
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

    def save_group(self, directory: str, test_name: str, key: Optional[str] = None):
        """ Saves the results of a single group to a specified directory as a pickle file. 
        The group to be saved can be specified by the 'key' parameter; if not provided, 
        the current group is used. The file is named using the test name and the current group key.

        Args:
            directory [str]: The directory in which to save the results.
            test_name [str]: The name of this specific test, used in the output filename.
            key [Optional[str]]: The group key to use. If not specified, the current group is used.

        Return:
            None. The function saves the group metrics to a file.
        """
        if key is None:
            group = self.current_group
        else:
            group = self.groups[key]
        save_path = Path(directory) / f"{test_name}_{self.current_group_key}.pkl"
        print(f"Saving group metrics to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(group, f)

    def save(self, directory: str, test_name: str):
        """ Save all metric groups to a pickle file.

        Args:
            directory [str]: The directory in which to save the results.
            test_name [str]: The test name, used as the file name prefix.

        Return:
            None. The metric groups are saved to a file in the specified directory.
        """
        save_path = Path(directory) / f"{test_name}_metrics.pkl"
        print(f"Metrics will save to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(self.groups, f)

    def print_group_metrics(self, key: Optional[str] = None):
        """ Prints the metrics for a specified group. If no group key is provided, it uses the current group.

        Args:
            key [Optional[str]]: The group key to select which group's metrics to print. If None, uses the current group.

        Return:
            The result of print_metrics for the selected group.
        """
        if key is not None:
            self.current_group = self.groups[key]
            self.current_group_key = key
        return self.print_metrics(self.current_group)

    def print_overall_metrics(self):
        """ Aggregates metrics from all groups and prints the overall metrics.
        It collects all unique metric keys across groups, aggregates their values,
        and then prints the metrics for the combined results.

        Args:
            None

        Return:
            The result of the print_metrics function applied to the aggregated metrics (supergroup).
        """
        supergroup = {}
        keys = set()
        for group in self.groups.values():
            for key in group.keys():
                keys.add(key)

        for key in keys:
            metrics = []
            for group in self.groups.values():
                metrics.extend(group.get(key, []))
            supergroup[key] = metrics
        return self.print_metrics(supergroup)
