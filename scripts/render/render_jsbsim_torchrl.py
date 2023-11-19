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
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from scripts.train.torchrl_utils import (
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)


@hydra.main(version_base="1.1", config_path=".", config_name="torchrl_render_config")
def main(cfg: DictConfig):  # noqa: F821
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.exp_name)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)

    load_dir = '/home/dennismalmgren/repos/CloseAirCombat/pretrained/2023-11-18/lowlevel'
    load_state = torch.load(f"{load_dir}/training_snapshot_3000000.pt")
    model.load_state_dict(load_state['model'])
    
    #
    os.mkdir('runs')
    run_dir = 'runs'
    # Main loop
    start_time = time.time()
    with set_exploration_type(ExplorationType.MODE):
        #logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_td = eval_env.reset()
        done = False
        eval_env.render(mode='txt', filepath=f'{run_dir}/{exp_name}.txt.acmi')
        while not done:
            render_td = exploration_policy(render_td)
            render_td = eval_env.step(render_td)
            render_episode_rewards += render_td["next", "reward"].sum(-2).item()
            eval_env.render(mode='txt', filepath=f'{run_dir}/{exp_name}.txt.acmi')
            done = render_td["next", "done"].all()


    total_time = time.time() - start_time

    print(f"Rendering took {total_time:.2f} seconds to finish")
    print(f"Average reward: {render_episode_rewards:.2f}")

if __name__ == "__main__":
    main()