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

from scripts.train.opus_utils_p3o import (
    make_environment,
    make_ppo_models,
    eval_model
)

@hydra.main(version_base="1.1", config_path=".", config_name="opus_render_p3o")
def main(cfg: DictConfig):  # noqa: F821
    device = torch.device("cpu")

    torch.manual_seed(cfg.random.seed)
    np.random.seed(cfg.random.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    policy_module, value_module = make_ppo_models(cfg, eval_env, device)
    actor = policy_module
    critic = value_module

    run_as_debug = False
    load_from_saved_models = False
    load_from_debug = False
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    if run_as_debug:
        if load_from_debug:
            outputs_folder = "../../"
        elif load_from_saved_models:
            outputs_folder = "../../../scripts/train/saved_models/"
        else:
            outputs_folder = "../../../scripts/train/outputs/"
    else:
        if load_from_debug:
            outputs_folder = "../../../../../outputs"
        else:
            outputs_folder = "../../"
    model_name = "training_snapshot"
    if load_from_saved_models:
        model_name = "training_snapshot_waypoint"
    if load_from_saved_models:
        run_id = ""
    else:
        run_id = "2024-03-26/00-30-10/"
    iteration = 10408000
    model_load_filename = f"{model_name}_{iteration}.pt"
    load_model_dir = outputs_folder + run_id
    print('Loading model from ' + load_model_dir)
    loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
    actor_state = loaded_state['model_actor']
    critic_state = loaded_state['model_critic']
    
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
   
    exp_name = generate_exp_name("OPUS_Render", cfg.logger.exp_name)
    os.mkdir('runs')
    run_dir = 'runs'
    # Main loop
    start_time = time.time()
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
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