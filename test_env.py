from envs.JSBSim.envs.selfplay_env import SelfPlayEnv
import numpy as np
import pdb
import time
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.flatten_utils import DictFlattener


def test_env():
    # env = SelfPlayEnv(config='selfplay_task.yaml')
    env = SelfPlayEnv(config='selfplay_with_missile_task.yaml')
    act_space = env.action_space
    act_flattener = DictFlattener(act_space)

    env.reset()
    cur_step = -1
    reward_blue, reward_red = 0., 0.
    start_time = time.time()
    while True:
        cur_step += 1
        # flying straight forward
        actions = {"red_fighter": np.array([20., 18.6, 20., 0.]), 'blue_fighter': np.array([20., 18.6, 20., 0.])}
        # random fly
        # actions = {"red_fighter": act_flattener(act_space.sample()), 'blue_fighter': act_flattener(act_space.sample())}
        next_obs, reward, done, env_info = env.step(actions)
        reward_blue += reward['blue_fighter']
        reward_red += reward['red_fighter']
        print(reward_blue, reward_red)
        if done:
            print(env_info)
            break
    print(time.time() - start_time)


def test_parallel_env():
    
    def make_train_env(num_env):
        return SubprocVecEnv([SelfPlayEnv for _ in range(num_env)])

    start_time = time.time()
    num_env = 10
    envs = make_train_env(num_env)
    act_space = envs.action_space
    act_flattener = DictFlattener(act_space)

    n_total_steps = 50000
    n_current_steps = 0
    n_current_episodes = 0
    obss = envs.reset()
    while n_current_steps < n_total_steps:
        actions = [{"red_fighter": act_flattener(act_space.sample()), 'blue_fighter': act_flattener(act_space.sample())} for _ in range(num_env)]
        next_obss, rewards, dones, env_infos = envs.step(actions)
        new_samples = list(zip(obss, actions, rewards, next_obss, dones))
        n_current_steps += len(new_samples)
        for i, done in enumerate(dones):
            if done:
                n_current_episodes += 1
                # print(env_infos[i])
    print(f"Collect data finish: total step {n_current_steps}, total episode {n_current_episodes}, timecost: {time.time() - start_time:.2f}s")
    envs.close()


test_env()
# test_parallel_env()