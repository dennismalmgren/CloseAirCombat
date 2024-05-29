import gymnasium
from gymnasium import spaces
import numpy as np
from typing import Any

class CustomContinuousEnv(gymnasium.Env):
    def __init__(self, target_point=0.0):
        super(CustomContinuousEnv, self).__init__()
        
        # Define the action space: continuous action in range [-1, 1]
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        
        # Define the observation space: continuous observation, always zero
        self.observation_space = spaces.Box(low=np.array([-np.inf]), high=np.array([np.inf]), shape=(1,), dtype=np.float32)
        
        # Target point for the reward
        self.target_point = target_point

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        # Return the initial observation (always zero)
        return np.zeros(1, dtype=np.float32), {}
    
    def step(self, action):
        action = action[0]  # Extract the action value from the action array
        reward = max(0.0, 1.0 - abs(action - self.target_point))
        
        # Observation is always zero
        observation = np.zeros(1, dtype=np.float32)
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass

# Register the custom environment with OpenAI Gym
gymnasium.envs.registration.register(
    id='CustomContinuousEnv-v0',
    entry_point=CustomContinuousEnv,
    max_episode_steps=100,
)

# Usage example
if __name__ == "__main__":
    env = gymnasium.make('CustomContinuousEnv-v0', target_point=0.5)
    obs = env.reset()
    action = np.array([0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {reward}, Done: {terminated}, Info: {info}")