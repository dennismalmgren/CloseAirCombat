import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np
from ..utils.utils import LLA2NED, NED2LLA

class OpusWaypointReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', 
                                                                              '_wp_dist_r']]

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
        curriculum = env.curriculum
        active_task = int(agent.get_property_value(c.current_task_id))
        if active_task == 1:
            active_task_type = int(agent.get_property_value(c.task_1_type_id))
        else:
            active_task_type = int(agent.get_property_value(c.task_2_type_id))

        if active_task_type != 2:
            reward = wp_pos_x_r = wp_pos_y_r = wp_pos_z_r = wp_time = 0.0
            return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_time))
        else:
            active_waypoint_index = active_task - 1
            wp_time_r = 0
            wp_dist_r = 0

            agent_alt = agent.get_property_value(c.position_h_sl_m)                  # 0. altitude  (unit: m)
            agent_lat = agent.get_property_value(c.position_lat_geod_deg)            # 1. latitude geodetic (unit: deg)
            agent_lon = agent.get_property_value(c.position_long_gc_deg)           
            current_north, current_east, current_down = LLA2NED(agent_lat, agent_lon, agent_alt, env.task.lat0, env.task.lon0, env.task.alt0)

            dist = np.linalg.norm(np.array([current_north - curriculum.target_waypoints[active_waypoint_index][0], 
                             current_east - curriculum.target_waypoints[active_waypoint_index][1], 
                             current_down - curriculum.target_waypoints[active_waypoint_index][2]]))
            if dist < 100:
                wp_dist_r = 10

            #roll_error_scale = 0.35  # radians ~= 20 degrees
            #wp_roll_r = 0.001 * math.exp(-((env.agents[agent_id].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))
            current_speed = agent.get_property_value(c.velocities_u_mps)
            current_mach_limit = curriculum.target_mach_limits[active_waypoint_index]
            #wp_speed_r = 0.0
            #if current_speed/340.0 > current_mach_limit:
            #    wp_speed_r = -0.01  * math.exp(-(max(0, current_speed / 340.0 - current_mach_limit)) ** 2)

            reward = wp_dist_r #+ wp_time_r #+ wp_speed_r #+ wp_roll_r
            return self._process(reward, agent_id, (wp_dist_r, ))
    