import numpy as np
from gymnasium import spaces
from typing import List, Tuple
from abc import ABC, abstractmethod
from ..core.catalog import Catalog as c


class BaseTask(ABC):
    """
    Base Task class.
    A class to subclass in order to create a task with its own observation variables,
    action variables, termination conditions and reward functions.
    """
    def __init__(self, config):
        self.config = config
        self.reward_functions = []
        self.termination_conditions = []
        self.load_variables()
        self.load_observation_space()
        self.load_action_space()

    @property
    def num_agents(self):
        return 1

    @abstractmethod
    def load_variables(self):
        self.state_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,
            c.fcs_elevator_cmd_norm,
            c.fcs_rudder_cmd_norm,
            c.fcs_throttle_cmd_norm,
        ]

    @abstractmethod
    def load_observation_space(self):
        """
        Load observation space
        """
        self.observation_space = spaces.Discrete(5)

    @abstractmethod
    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = spaces.Discrete(5)

    def reset(self, env):
        """Task-specific reset

        Args:
            env: environment instance
        """
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)
            
    def step(self, env):
        """ Task-specific step

        Args:
            env: environment instance
        """
        pass

    def get_reward_keys(self):
        """Get reward keys

        Returns:
            (list): reward keys
        """
        return ["reward"] + [reward_item_name for reward_function in self.reward_functions for reward_item_name in reward_function.reward_item_names]
    
    def get_logged_reward_keys(self):
        """Get reward keys

        Returns:
            (list): reward keys
        """
        return ["reward"] + [reward_item_name for reward_function in self.logged_reward_functions for reward_item_name in reward_function.reward_item_names]
    
    def get_logged_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        reward = 0.0
        for reward_function in self.logged_reward_functions:
            func_reward = reward_function.get_reward(self, env, agent_id)
            reward += func_reward
        return reward, info

    def get_reward(self, env, agent_id, info={}) -> Tuple[float, dict]:
        """
        Aggregate reward functions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                reward(float): total reward of the current timestep
                info(dict): additional info
        """
        reward = 0.0
        for reward_function in self.reward_functions:
            func_reward = reward_function.get_reward(self, env, agent_id)
            reward += func_reward
        return reward, info
    
    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """
        Aggregate termination conditions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                done(bool): whether the episode has terminated
                info(dict): additional info
        """
        done = False
        success = True
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s
            if done:
                break
        return done, info

    def get_obs(self, env, agent_id):
        """Extract useful informations from environment for specific agent_id.
        """
        return np.zeros(2)

    def normalize_action(self, env, agent_id, action):
        """Normalize action to be consistent with action space.
        """
        return np.array(action)

    def _convert_to_quaternion(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr
        return np.asarray([w, x, y, z])

    def _convert_to_sincos(self, angle):
        return np.array([np.sin(angle), np.cos(angle)])
    
    def constrain_angle_diff(self, angle_diff):
        """
        Constrain an angle difference to the range [-pi, pi].
        """
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi