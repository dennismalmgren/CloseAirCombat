import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c
import numpy as np

class OpusAltitudeReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_alt', '_roll']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        """
        task_variables = task.calculate_task_variables(env, agent_id)
        delta_altitude_m = task_variables[0]
        delta_phi_rad = task_variables[1]
        #check if c.attitude_roll_rad is equivalent

        alt_error_scale = 15.24  # m
        alt_r = math.exp(-((delta_altitude_m / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((delta_phi_rad / roll_error_scale) ** 2))
#        smoothness_variables = task.calculate_smoothness(env, agent_id)
#        smoothness_roll_scale = 5.0

#        smooth_roll_r = math.exp(-((smoothness_variables[3] / smoothness_roll_scale) ** 2))
        #reward = (1 / 2) * ((alt_r * roll_r) ** (1 / 2) + (smooth_roll_r))
        reward =(alt_r * roll_r) ** (1 / 2)
        #return self._process(reward, agent_id, (alt_r, roll_r, smooth_roll_r))
        return self._process(reward, agent_id, (alt_r, roll_r))
