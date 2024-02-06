import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.patrol_env import PatrolEnv

from gymnasium.wrappers.time_limit import TimeLimit

def main():
    env = PatrolEnv()
    env = TimeLimit(env, max_episode_steps=10000)
    obs, info = env.reset()
    action_mask = info['action_mask']
    done = False
    episode_reward = 0
    reward_scale = 0.01
    while not done:
        action = np.asarray(np.where(action_mask > 0)[0][0])
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated | truncated
        action_mask = info['action_mask']
        episode_reward += reward * reward_scale
    print(f'Episode reward: {episode_reward}')

if __name__=="__main__":
    main()


#what's next?
#show traversed path (visit count) in a render script.
#add intensities
#calculate distances to all nodes and store (cost basis)
#train for three cases of intensities. uniform, along one edge, and along two edges.
    