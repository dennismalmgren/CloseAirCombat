import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class OpusAltitudeSpeedHeadingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
    #    self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_alt', '_roll', '_speed']]

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
        heading_r = math.exp(-((delta_heading_deg / heading_error_scale) ** 2))

        alt_error_scale = 15.24  # m
        alt_r = math.exp(-((delta_altitude_m / alt_error_scale) ** 2))

        roll_error_scale = 0.35  # radians ~= 20 degrees
        roll_r = math.exp(-((delta_roll_rad / roll_error_scale) ** 2))

        speed_error_scale = 24  # mps (~10%)
        speed_r = math.exp(-((delta_speed / speed_error_scale) ** 2))

        #Ignore heading
        reward = (alt_r * roll_r * speed_r) ** (1 / 3)
        return self._process(reward, agent_id, (alt_r, roll_r, speed_r))
    
#        reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)
#        return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))
