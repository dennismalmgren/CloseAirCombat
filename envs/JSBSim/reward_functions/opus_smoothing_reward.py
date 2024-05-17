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
        #self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_speed', 
        #                                                                      '_smooth_p', '_smooth_q', '_smooth_r', 
        #                                                                      '_smooth_pdot', '_smooth_qdot', '_smooth_rdot']]
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_heading', '_alt', '_speed', 
                                                                            #  '_smooth_p', '_smooth_q', '_smooth_r', 
                                                                              '_smooth_pdot', '_smooth_qdot', '_smooth_rdot']]
    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable
      
        """
        task_variables = task.calculate_task_variables(env, agent_id)
        delta_altitude_m = task_variables[0]
        delta_speed = task_variables[1]
        delta_heading_rad = task_variables[2]
        delta_heading_deg = delta_heading_rad * 180 / math.pi

        heading_error_scale = 5.0  # degrees
        heading_error_tolerance = 5.0
        delta_heading_deg_with_tolerance = max(0, abs(delta_heading_deg) - heading_error_tolerance)
        heading_r = math.exp(-((delta_heading_deg_with_tolerance / heading_error_scale) ** 2))

        alt_error_scale = 15.24  # m
        altitude_error_tolerance = 100.24 # m
        delta_altitude_m_with_tolerance = max(0, abs(delta_altitude_m) - altitude_error_tolerance)
        alt_r = math.exp(-((delta_altitude_m_with_tolerance / alt_error_scale) ** 2))

        speed_error_scale = 24  # mps (~10%)
        speed_error_tolerance = 24 # mps
        delta_speed_with_tolerance = max(0, abs(delta_speed) - speed_error_tolerance)
        speed_r = math.exp(-((delta_speed_with_tolerance / speed_error_scale) ** 2))

        task_history_variables = task.get_task_history_variables(env, agent_id)

        p_history = task_history_variables[0]
        smoothness_value_p = np.abs(p_history[-1])
        q_history = task_history_variables[1]
        smoothness_value_q = np.abs(q_history[-1])
        r_history = task_history_variables[2]
        smoothness_value_r = np.abs(r_history[-1])

        pdot_history = task_history_variables[3]
        smoothness_value_pdot = np.abs(pdot_history[-1])
        qdot_history = task_history_variables[4]
        smoothness_value_qdot = np.abs(qdot_history[-1])
        rdot_history = task_history_variables[5]
        smoothness_value_rdot = np.abs(rdot_history[-1])

        #smoothness_p_scale = 0.1
        #smooth_p_r = math.exp(-((smoothness_value_p / smoothness_p_scale) ** 2))

        #smoothness_q_scale = 0.1
        #smooth_q_r = math.exp(-((smoothness_value_q / smoothness_q_scale) ** 2))

        #smoothness_r_scale = 0.1
        #smooth_r_r = math.exp(-((smoothness_value_r / smoothness_r_scale) ** 2))

        smoothness_pdot_scale = 1.0
        smooth_pdot_r = math.exp(-((smoothness_value_pdot / smoothness_pdot_scale) ** 2))
        smoothness_qdot_scale = 1.0
        smooth_qdot_r = math.exp(-((smoothness_value_qdot / smoothness_qdot_scale) ** 2))
        smoothness_rdot_scale = 1.0
        smooth_rdot_r = math.exp(-((smoothness_value_rdot / smoothness_rdot_scale) ** 2))
        
        #smoothness_reward = (smooth_p_r * smooth_q_r * smooth_r_r * smooth_pdot_r * smooth_qdot_r * smooth_rdot_r) ** (1 / 6)
        smoothness_reward = (smooth_pdot_r * smooth_qdot_r * smooth_rdot_r) ** (1 / 3)

        task_reward =  (heading_r * alt_r * speed_r) ** (1 / 3)
        if task_reward >= 1.0:
            reward = task_reward + smoothness_reward
        else:
            reward = task_reward
            smooth_p_r = smooth_q_r = smooth_r_r = smooth_pdot_r = smooth_qdot_r = smooth_rdot_r = 0.0

        return self._process(reward, agent_id, (heading_r, alt_r, speed_r, 
                                               # smooth_p_r, smooth_q_r, smooth_r_r, 
                                                smooth_pdot_r, smooth_qdot_r, smooth_rdot_r))
