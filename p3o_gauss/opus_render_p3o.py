# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""

import time
import sys
import os

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from torchrl.record.loggers import generate_exp_name, get_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from objectives import P3OLoss

from .opus_utils_p3o import (
    make_environment,
    make_agent,
    eval_model,
    load_model_state,
    load_observation_statistics
)

@hydra.main(version_base="1.1", config_path=".", config_name="opus_train_p3o")
def main(cfg: DictConfig):  # noqa: F821
    device = torch.device("cpu")

    torch.manual_seed(cfg.random.seed)
    np.random.seed(cfg.random.seed)

    # Create environments
    cfg.collector.env_per_collector = 1
    train_env, eval_env = make_environment(cfg)
   #eval_env.eval()
    reward_keys = list(train_env.reward_spec.keys())

    # Create agent
    policy_module, value_module, support = make_agent(cfg, eval_env, device)
    actor = policy_module
    critic = value_module

    load_model = True
    if load_model:
        #Ione, smoothing+curriculum
        model_dir="2024-05-20/11-33-18/"
        model_name = "training_snapshot_40048000"
        observation_statistics_name = "observation_statistics_40048000" 

        #Ione, no smoothing, no curriculum
        model_dir="2024-10-21/08-42-40/"
        model_name = "training_snapshot_40528000"
        observation_statistics_name = "observation_statistics_40528000" 

        #Zoe, smoothing, no curriculum
        model_dir="2024-10-22/00-12-02/"
        model_name = "training_snapshot_32048000"
        observation_statistics_name = "observation_statistics_32048000" 

        loaded_state = load_model_state(model_name, model_dir)
        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)
        observation_statistics = load_observation_statistics(observation_statistics_name, model_dir)
        eval_env.transform[3]._td = observation_statistics.to(device)
    exp_name = generate_exp_name("OPUS_Render", cfg.logger.exp_name)
    os.mkdir('runs')
    run_dir = 'runs'
    # Main loop
    start_time = time.time()
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
    #with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        #logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_td = eval_env.reset()
        done = False
        eval_env.render(mode='txt', filepath=f'{run_dir}/{exp_name}.txt.acmi')
        action_count = 0
        while not done:
            render_td = actor(render_td)
            render_td = eval_env.step(render_td)
            render_episode_rewards += render_td["next", "reward"].sum(-2).item()
            eval_env.render(mode='txt', filepath=f'{run_dir}/{exp_name}.txt.acmi')
            done = render_td["next", "done"].all()
            render_td = step_mdp(render_td)
            action_count += 1

    total_time = time.time() - start_time

    print(f"Rendering took {total_time:.2f} seconds to finish")
    print(f"Average reward: {render_episode_rewards:.2f}")
    print(f'Episode length: {action_count}')


if __name__ == "__main__":
    main()