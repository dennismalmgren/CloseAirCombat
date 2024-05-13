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
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.collectors import SyncDataCollector

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


@hydra.main(version_base="1.1", config_path=".", config_name="torchrl_cont_config_missile")
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

    # Create agent
    model, lowlevelpolicy = make_sac_agent(contcontrolcfg, train_env, eval_env, device)

    load_dir = '/home/dennismalmgren/repos/CloseAirCombat/pretrained/2023-11-18/lowlevel'
    load_state = torch.load(f"{load_dir}/training_snapshot_3000000.pt")
    model.load_state_dict(load_state['model'])

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode},
        )

    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size
    total_network_updates = (
        (cfg.collector.total_frames // cfg.collector.frames_per_batch)
        * cfg.loss.ppo_epochs
        * num_mini_batches
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    def make_env(cfg, lowlevelpolicy):
        def maker():
            env = env_maker(cfg, lowlevelpolicy)
            train_env = apply_env_transforms(env)
            return train_env
        return maker
    train_env, eval_env = make_ppo_environment(cfg, lowlevelpolicy)

    # Create agent

    actor, critic = make_ppo_models(cfg, eval_env)
    actor, critic = actor.to(device), critic.to(device)
    
    
    # Create SAC loss
    #loss_module, target_net_updater = make_loss_module(cfg, model)
    
    # Create off-policy collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg, lowlevelpolicy),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    sampler = SamplerWithoutReplacement()

    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizers
    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr)

    #load_dir = '/home/dennismalmgren/repos/CloseAirCombat/scripts/train/outputs/2023-11-18/03-57-26'
    #load_state = torch.load(f"{load_dir}/training_snapshot_1000000.pt")
#    load_dir = '/home/dennismalmgren/repos/CloseAirCombat/scripts/train/outputs/2023-11-17/08-05-57'
#    load_state = torch.load(f"{load_dir}/training_snapshot_1000000.pt")
    #model.load_state_dict(load_state['model'])
    #loss_module.load_state_dict(load_state['loss'])
    #optimizer_alpha.load_state_dict(load_state['optimizer_alpha'])
    #optimizer_critic.load_state_dict(load_state['optimizer_critic'])
    #optimizer_actor.load_state_dict(load_state['optimizer_actor'])
    #load_dir = '/home/dennismalmgren/repos/CloseAirCombat/pretrained/2023-11-18/lowlevel'
    #load_dir = '/home/dennismalmgren/repos/CloseAirCombat/scripts/train/outputs/2023-11-18/22-57-45'
    #load_state = torch.load(f"{load_dir}/training_snapshot_3000000.pt")
    #model.load_state_dict(load_state['model'])
    #loss_module.load_state_dict(load_state['loss'])
    #optimizer_alpha.load_state_dict(load_state['optimizer_alpha'])
    #optimizer_critic.load_state_dict(load_state['optimizer_critic'])
    #optimizer_actor.load_state_dict(load_state['optimizer_actor'])

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_eval_iter = cfg.logger.eval_iter

    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])
    sampling_start = time.time()
    for i, tensordict in enumerate(collector):
        
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = tensordict.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)
        
        episode_rewards = tensordict["next", "episode_reward"][tensordict["next", "terminated"]]
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][tensordict["next", "terminated"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )
        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):
            with torch.no_grad():
                tensordict = adv_module(tensordict)
            tensordict = tensordict.reshape(-1)

            data_buffer.extend(tensordict)

            for k, batch in enumerate(data_buffer):
                batch = batch.to(device)

                num_network_updates += 1

                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                actor_optim.zero_grad() 
                critic_optim.zero_grad()
                actor_loss.backward()
                critic_loss.backward()

                actor_optim.step()
                critic_optim.step()

        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})

        log_info.update(
            {
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if ((i - 1) * frames_in_batch) // cfg_logger_eval_iter < (
                i * frames_in_batch
            ) // cfg_logger_eval_iter:
                actor.eval()
                eval_start = time.time()
                test_rewards = eval_ppo_model(
                    actor, eval_env, num_episodes=5
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_rewards.mean(),
                        "eval/time": eval_time,
                    }
                )
                actor.train()

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()
    
        if abs(collected_frames % cfg.train.save_interval) < frames_per_batch:
            savestate = {
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                        'loss': loss_module.state_dict(),                  
                        'optimizer_critic': critic_optim.state_dict(),
                        'optimizer_actor': actor_optim.state_dict(),
            }
            torch.save(savestate, f"training_snapshot_{collected_frames}.pt")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")

if __name__ == "__main__":
    main()