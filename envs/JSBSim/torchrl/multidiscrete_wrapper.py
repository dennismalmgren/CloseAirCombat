import gymnasium as gym
from gymnasium.spaces import Discrete
import numpy as np

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete), "Action space must be MultiDiscrete"
        self.original_action_space = env.action_space
        self.action_space = Discrete(np.prod(self.original_action_space.nvec))
        #2 000 000 actions. is it even feasible?
        #lets go for continuous actions.
        # Create a mapping from discrete actions to MultiDiscrete actions
        self.action_mapping = list(np.ndindex(*self.original_action_space.nvec))
