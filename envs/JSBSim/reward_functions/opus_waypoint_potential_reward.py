import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np

class OpusWaypointPotentialReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', 
                                                                              '_wp_approach']]
        self.is_potential = True

    def reset(self, task, env):
        self.time_taken = 0

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
                    # task type id
        # 0: no mission (not used),
        # 1: travel in heading at altitude and speed
        # 2: travel to waypoint
        # 3: search area
        # 4: engage target
        """
        agent = env.agents[agent_id]
        active_task = int(agent.get_property_value(c.current_task_id))
        if active_task == 1:
            active_task_type = int(agent.get_property_value(c.task_1_type_id))
        else:
            active_task_type = int(agent.get_property_value(c.task_2_type_id))
        if active_task_type != 2:
            reward = _wp_approach = 0.0
            return self._process(reward, agent_id, (_wp_approach))
        else:
            active_waypoint_index = active_task - 1
            wp_time_r = -1
            wp_dist = 0
            current_north = agent.get_property_value(c.position_north_m)
            current_east = agent.get_property_value(c.position_east_m)
            current_down = agent.get_property_value(c.position_down_m)
            dist = np.linalg.norm(np.array([current_north - task.waypoints[active_waypoint_index][0], 
                             current_east - task.waypoints[active_waypoint_index][1], 
                             current_down - task.waypoints[active_waypoint_index][2]]))
            dist_error_scale = 15.24  # m
            alt_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_altitude) / alt_error_scale) ** 2))
 
            #reach if within 200m.
            if np.linalg.norm(np.array([current_north - self.current_waypoint[0], 
                             current_east - self.current_waypoint[1]])) < 100 and \
                                np.norm(np.array([current_down - self.current_waypoint[2]])) < 30:
                wp_dist = 1
                
            reward = wp_dist + wp_time_r
            return self._process(reward, agent_id, (wp_dist, wp_time_r))
    