import math
import numpy as np
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class OpusSmoothingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
    #    self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed', '_smoothness']]
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed', '_smoothness_p', '_smoothness_w', '_smoothness_p_ref', 'smoothness_w_ref', '_value_w_r', '_value_p_r']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

      
        """
        task_variables = task.calculate_task_variables(env, agent_id)
        delta_altitude_m = task_variables[0]
        delta_roll_rad = task_variables[1]
        delta_speed = task_variables[2]
        delta_heading_rad = task_variables[3]
        delta_heading_deg = delta_heading_rad * 180 / math.pi

        heading_error_scale = 5.0  # degrees
        heading_error_tolerance = 5.0
        delta_heading_deg_with_tolerance = max(0, abs(delta_heading_deg) - heading_error_tolerance)

        heading_r = math.exp(-((delta_heading_deg_with_tolerance / heading_error_scale) ** 2))

        alt_error_scale = 15.24  # m
        altitude_error_tolerance = 100.24 # m
        delta_altitude_m_with_tolerance = max(0, abs(delta_altitude_m) - altitude_error_tolerance)
        alt_r = math.exp(-((delta_altitude_m_with_tolerance / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_error_tolerance = 0.35 # radians
        delta_roll_with_tolerance = max(0, abs(delta_roll_rad) - roll_error_tolerance)

        roll_r = math.exp(-((delta_roll_with_tolerance / roll_error_scale) ** 2))

        speed_error_scale = 24  # mps (~10%)
        speed_error_tolerance = 24 # mps
        delta_speed_with_tolerance = max(0, abs(delta_speed) - speed_error_tolerance)
        speed_r = math.exp(-((delta_speed_with_tolerance / speed_error_scale) ** 2))
        #move to its own reward.
       # smoothness_variables = task.calculate_smoothness(env, agent_id)
       # smoothness_r = 0.000001 * np.sum(smoothness_variables**2)
        #Ignore heading
        #reward = (alt_r * roll_r * speed_r) ** (1 / 3)
        #return self._process(reward, agent_id, (alt_r, roll_r, speed_r))

        smoothness_variables = task.get_smoothness_variables(env, agent_id)
        smoothness_p = smoothness_variables[3]
        smoothness_value_p = np.sum(np.abs(smoothness_p[-1] - smoothness_p[-2]))

        smoothness_p_scale = 5.0
        smooth_roll_r = math.exp(-((smoothness_value_p / smoothness_p_scale) ** 2))
        smooth_p_ref = smooth_roll_r

        smoothness_w = smoothness_variables[2]
        smoothness_value_w = np.sum(np.abs(smoothness_w[-1] - smoothness_w[-2]))

        smoothness_w_scale = 10.0
        smooth_w_r = math.exp(-((smoothness_value_w / smoothness_w_scale) ** 2))
        smooth_w_ref = smooth_w_r

        task_history = task.get_task_history_variables(env, agent_id)
        
        w_history = task_history[0]
        value_w = np.sum(np.abs(w_history[-1] - w_history[-2]))
        w_r_scale = 0.05

        value_w_r = math.exp(-((value_w / w_r_scale) ** 2))

        p_history = task_history[1]
        p_history_cos = np.cos(p_history)
        p_history_sin = np.sin(p_history)
        value_p_1 = np.sum(np.abs(p_history_sin[-1] - p_history_sin[-2]))
        value_p_2 = np.sum(np.abs(p_history_cos[-1] - p_history_cos[-2]))

        p_r_scale = 0.09
        value_p_r = 0.5 * math.exp(-((value_p_1 / p_r_scale) ** 2)) + 0.5 * math.exp(-((value_p_2 / p_r_scale) ** 2))

        #* smooth_roll_r * smooth_w_r
        reward =  (heading_r * alt_r * roll_r * speed_r) ** (1 / 4) #- smoothness_r
        if (heading_r * alt_r * roll_r * speed_r) >= 1.0:
            reward += (smooth_roll_r * smooth_w_r * value_w_r * value_p_r)**(1 / 4)
        else:
            smooth_roll_r = smooth_w_r = 0.0

       # return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r, smoothness_r))
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r, smooth_roll_r, smooth_w_r, smooth_p_ref, smooth_w_ref, value_w_r, value_p_r))
