import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class OpusWaypointReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', 
                                                                              '_wp_pos_x', '_wp_pos_y', '_wp_pos_z', 
                                                                              '_wp_vel_x, _wp_vel_y, _wp_vel_z', 
                                                                              '_wp_time']]

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
            reward = wp_pos_x_r = wp_pos_y_r = wp_pos_z_r = wp_vel_x_r = wp_vel_y_r = wp_vel_z_r = wp_time_r = 0.0
            return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_vel_x_r, wp_vel_y_r, wp_vel_z_r, wp_time_r))
        else:
            #first, check which part of the waypoint should be the active one.
            delta_1_time = task.delta_1_time
            delta_2_time = task.delta_2_time
            if delta_1_time >= 0:
                if delta_1_time >= 2.0: #no reward until the final 5 seconds
                    reward = wp_pos_x_r = wp_pos_y_r = wp_pos_z_r = wp_vel_x_r = wp_vel_y_r = wp_vel_z_r = wp_time_r = 0.0
                    return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_vel_x_r, wp_vel_y_r, wp_vel_z_r, wp_time_r))
                else:
                    #multiply by a time reward that's exponential in the time remaining.
                    wp_time_r = math.exp(-((delta_1_time / 2.0) ** 2))

                    pos_error_scale = 15.24  # m
                    wp_pos_x_r = math.exp(-((task.delta_1_north / pos_error_scale) ** 2)) * wp_time_r
                    wp_pos_y_r = math.exp(-((task.delta_1_east / pos_error_scale) ** 2)) * wp_time_r
                    wp_pos_z_r = math.exp(-((task.delta_1_down / pos_error_scale) ** 2)) * wp_time_r
                    vel_error_scale = 15 # mps
                    wp_vel_x_r = math.exp(-((task.delta_1_v_north / vel_error_scale) ** 2)) * wp_time_r
                    wp_vel_y_r = math.exp(-((task.delta_1_v_east / vel_error_scale) ** 2)) * wp_time_r
                    wp_vel_z_r = math.exp(-((task.delta_1_v_down / vel_error_scale) ** 2)) * wp_time_r
                    reward = (wp_pos_x_r * wp_pos_y_r * wp_pos_z_r * wp_vel_x_r * wp_vel_y_r * wp_vel_z_r * wp_time_r) ** (1 / 7)
                    return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_vel_x_r, wp_vel_y_r, wp_vel_z_r, wp_time_r))
            else:
                if delta_2_time >= 0:
                    if delta_2_time >= 2.0:
                        reward = wp_pos_x_r = wp_pos_y_r = wp_pos_z_r = wp_vel_x_r = wp_vel_y_r = wp_vel_z_r = wp_time_r = 0.0
                        return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_vel_x_r, wp_vel_y_r, wp_vel_z_r, wp_time_r))
                    else:
                        #multiply by a time reward that's exponential in the time remaining.
                        wp_time_r = math.exp(-((delta_2_time / 2.0) ** 2))

                        pos_error_scale = 15.24
                        wp_pos_x_r = math.exp(-((task.delta_2_north / pos_error_scale) ** 2)) * wp_time_r
                        wp_pos_y_r = math.exp(-((task.delta_2_east / pos_error_scale) ** 2)) * wp_time_r
                        wp_pos_z_r = math.exp(-((task.delta_2_down / pos_error_scale) ** 2)) * wp_time_r
                        vel_error_scale = 15 # mps
                        wp_vel_x_r = math.exp(-((task.delta_2_v_north / vel_error_scale) ** 2)) * wp_time_r
                        wp_vel_y_r = math.exp(-((task.delta_2_v_east / vel_error_scale) ** 2)) * wp_time_r
                        wp_vel_z_r = math.exp(-((task.delta_2_v_down / vel_error_scale) ** 2)) * wp_time_r
                        reward = (wp_pos_x_r * wp_pos_y_r * wp_pos_z_r * wp_vel_x_r * wp_vel_y_r * wp_vel_z_r * wp_time_r) ** (1 / 7)
                        return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_vel_x_r, wp_vel_y_r, wp_vel_z_r, wp_time_r))
                else:
                    reward = wp_pos_x_r = wp_pos_y_r = wp_pos_z_r = wp_vel_x_r = wp_vel_y_r = wp_vel_z_r = wp_time_r = 0.0
                    return self._process(reward, agent_id, (wp_pos_x_r, wp_pos_y_r, wp_pos_z_r, wp_vel_x_r, wp_vel_y_r, wp_vel_z_r, wp_time_r))
