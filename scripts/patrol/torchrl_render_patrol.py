import sys
import os
import numpy as np
import torch
from torchrl.envs import GymWrapper, TransformedEnv, RewardSum, StepCounter, Compose, default_info_dict_reader, RewardScaling, step_mdp
from torchrl.collectors import RandomPolicy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.patrol_env import PatrolEnv

from gymnasium.wrappers.time_limit import TimeLimit

def main():
    env = PatrolEnv()
    env = GymWrapper(env, categorical_action_encoding=True)
    env.set_info_dict_reader(default_info_dict_reader(["action_mask"]))
    env = TransformedEnv(env,
                         Compose(
                             RewardSum(),
                             StepCounter(max_steps=10000),
                             RewardScaling(loc=0, scale=0.01)
                         ))
    policy = RandomPolicy(env.action_spec)
    episode = []
    td = env.reset()
    done = torch.any(td['done'])
    while not done:
        env.action_spec.update_mask(td['action_mask'])
        action = env.action_spec.rand()
        td['action'] = action
        td = env.step(td)
        hist = env.render()
        hist = hist[1]
        episode.append(td)
        td = step_mdp(td)
        done = torch.any(td['done'])
    episode_td = torch.stack(episode).to_tensordict()
    episode_reward = episode_td['episode_reward'][-1].item()
    print(f'Episode reward: {episode_reward}')

if __name__=="__main__":
    main()


#what's next?
#show traversed path (visit count) in a render script.
#add intensities
#calculate distances to all nodes and store (cost basis)
#train for three cases of intensities. uniform, along one edge, and along two edges.
    