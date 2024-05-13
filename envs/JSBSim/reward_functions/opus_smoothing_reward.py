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
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed', '_smoothness']]

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
        heading_error_tolerance = 2.5
        delta_heading_deg_with_tolerance = max(0, abs(delta_heading_deg) - heading_error_tolerance)

        heading_r = math.exp(-((delta_heading_deg_with_tolerance / heading_error_scale) ** 2))

        alt_error_scale = 15.24  # m
        altitude_error_tolerance = 7.12 # m
        delta_altitude_m_with_tolerance = max(0, abs(delta_altitude_m) - altitude_error_tolerance)
        alt_r = math.exp(-((delta_altitude_m_with_tolerance / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((delta_roll_rad / roll_error_scale) ** 2))

        speed_error_scale = 24  # mps (~10%)
        speed_r = math.exp(-((delta_speed / speed_error_scale) ** 2))
        #move to its own reward.
       # smoothness_variables = task.calculate_smoothness(env, agent_id)
       # smoothness_r = 0.000001 * np.sum(smoothness_variables**2)
        #Ignore heading
        #reward = (alt_r * roll_r * speed_r) ** (1 / 3)
        #return self._process(reward, agent_id, (alt_r, roll_r, speed_r))
        smoothness_variables = task.calculate_smoothness(env, agent_id)
        smoothness_roll_scale = 5.0

        smooth_roll_r = math.exp(-((smoothness_variables[3] / smoothness_roll_scale) ** 2))
        reward = (heading_r * alt_r * roll_r * speed_r * smooth_roll_r) ** (1 / 4) #- smoothness_r
       # return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r, smoothness_r))
        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r, smooth_roll_r))
