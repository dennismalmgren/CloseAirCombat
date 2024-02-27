import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class OpusHeadingReward(BaseRewardFunction):
    """
    Measure the difference between the current heading and the target heading
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_roll', '_speed']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance
        # task type id
        # 0: no mission (not used),
        # 1: travel in heading at altitude and speed
        # 2: travel to waypoint
        # 3: search area
        # 4: engage target
        Returns:
            (float): reward
        """
        #check that a heading task is active.
        active_task = int(env.agents[agent_id].get_property_value(c.current_task_id))
        if active_task == 1:
            active_task_type = int(env.agents[agent_id].get_property_value(c.task_1_type_id))
        else:
            active_task_type = int(env.agents[agent_id].get_property_value(c.task_2_type_id))
        if active_task_type != 1:
            reward = heading_r = alt_r = roll_r = speed_r = 0.0
            return self._process(0.0, agent_id, (heading_r, alt_r, roll_r, speed_r))
        else:
            delta_heading = task.delta_heading
            delta_altitude = task.delta_altitude
            delta_speed = task.delta_speed
            delta_heading_rad = math.atan2(delta_heading[0], delta_heading[1])
            delta_heading_deg = math.degrees(delta_heading_rad)

            heading_error_scale = 5.0  # degrees
            heading_r = math.exp(-((delta_heading_deg / heading_error_scale) ** 2))

            alt_error_scale = 15.24  # m
            alt_r = math.exp(-((delta_altitude[0] / alt_error_scale) ** 2))

            roll_error_scale = 0.35  # radians ~= 20 degrees
            roll_r = math.exp(-((env.agents[agent_id].get_property_value(c.attitude_roll_rad) / roll_error_scale) ** 2))

            speed_error_scale = 24  # mps (~10%)
            speed_r = math.exp(-((delta_speed[0] / speed_error_scale) ** 2))

            reward = (heading_r * alt_r * roll_r * speed_r) ** (1 / 4)
            return self._process(reward, agent_id, (heading_r, alt_r, roll_r, speed_r))
