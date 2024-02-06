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
import omegaconf
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from torchrl.record.loggers import generate_exp_name, get_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from scripts.train.torchrl_utils import env_maker, apply_env_transforms

from scripts.train.torchrl_utils import (
    log_metrics,
    make_collector,
    make_ppo_collector,
    make_environment,
    make_loss_module,
    make_ppo_environment,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
    make_ppo_models,
    eval_ppo_model
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
    contcontrolcfg = omegaconf.OmegaConf.load('/home/dennismalmgren/repos/CloseAirCombat/scripts/train/torchrl_cont_config.yaml')
    # Create environments
    train_env, eval_env = make_environment(contcontrolcfg)

    model, lowlevelpolicy = make_sac_agent(contcontrolcfg, train_env, eval_env, device)
    load_dir = '/home/dennismalmgren/repos/CloseAirCombat/pretrained/2023-11-18/lowlevel'
    load_state = torch.load(f"{load_dir}/training_snapshot_3000000.pt")
    model.load_state_dict(load_state['model'])

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.exp_name)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    def make_env(cfg, lowlevelpolicy):
        def maker():
            env = env_maker(cfg, lowlevelpolicy)
            train_env = apply_env_transforms(env)
            return train_env
        return maker
    
    train_env, eval_env = make_ppo_environment(cfg, lowlevelpolicy)
    actor, critic = make_ppo_models(cfg, eval_env)
    actor, critic = actor.to(device), critic.to(device)
    
    # Create agent
    load_dir = '/home/dennismalmgren/repos/CloseAirCombat/pretrained/2023-11-20/midlevel'
    #load_dir = '/home/dennismalmgren/repos/CloseAirCombat/scripts/train/outputs/2023-11-18/22-57-45'
    load_state = torch.load(f"{load_dir}/training_snapshot_645120.pt")
    actor.load_state_dict(load_state['actor'])
    critic.load_state_dict(load_state['critic'])


    eval_env.set_seed(cfg.seed)
    
    os.mkdir('runs')
    run_dir = 'runs'
    # Main loop
    start_time = time.time()
    with set_exploration_type(ExplorationType.RANDOM), torch.no_grad():
        #logging.info("\nStart render ...")
        render_episode_rewards = 0
        render_td = eval_env.reset()
        done = False
        eval_env.render(mode='txt', filepath=f'{run_dir}/{exp_name}.txt.acmi')
        action_count = 0
        while not done:
            render_td = actor(render_td)
            render_td = eval_env.step(render_td)
            render_episode_rewards += render_td["next", "reward"].sum(-1).item() #todo. should be sum(-2)
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