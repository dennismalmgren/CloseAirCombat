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
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from torchrl.record.loggers import generate_exp_name, get_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from objectives import P3OLoss

from scripts.train.opus_utils_p3o import (
    make_environment,
    make_ppo_models,
    eval_model
)

@hydra.main(version_base="1.1", config_path=".", config_name="opus_train_p3o")
def main(cfg: DictConfig):  # noqa: F821
    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        and torch.cuda.device_count() > 0
        and cfg.device == "cuda:0"
        else torch.device("cpu")
    )

    torch.manual_seed(cfg.random.seed)
    np.random.seed(cfg.random.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    policy_module, value_module = make_ppo_models(cfg, eval_env, device)
    actor = policy_module
    critic = value_module

   
    loss_module = P3OLoss(
        actor_network=actor,
        critic_network=critic,
        #clip_epsilon=cfg.optim.clip_epsilon,
        loss_critic_type=cfg.optim.loss_critic_type,
        entropy_coef=cfg.optim.entropy_coef,
        critic_coef=cfg.optim.critic_coef,
        normalize_advantage=True,
    )

    actor_optim = torch.optim.Adam(actor.parameters(), lr=cfg.optim.lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=cfg.optim.lr, eps=1e-5)


    load_model = True
    run_as_debug = False
    load_from_saved_models = False
    load_from_debug = False
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    if load_model:
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
                outputs_folder = "../../../../../outputs/"
            elif load_from_saved_models:
                outputs_folder = "../../../saved_models/"
            else:
                outputs_folder = "../../"
        model_name = "training_snapshot"
        if load_from_saved_models:
            model_name = "training_snapshot_heading"
        if load_from_saved_models:
            run_id = ""
        else:
            run_id = "2024-03-26/00-30-10/"
        iteration = 13712000
        model_load_filename = f"{model_name}_{iteration}.pt"
        load_model_dir = outputs_folder + run_id
        print('Loading model from ' + load_model_dir)
        loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
        actor_state = loaded_state['model_actor']
        critic_state = loaded_state['model_critic']
        actor_optim_state = loaded_state['actor_optimizer']
        critic_optim_state = loaded_state['critic_optimizer']
        collected_frames = loaded_state['collected_frames']['collected_frames']
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)
        actor_optim.load_state_dict(actor_optim_state)
        critic_optim.load_state_dict(critic_optim_state)
    else:          
        collected_frames = 0
    frames_remaining = cfg.collector.total_frames - collected_frames

 # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_environment(cfg, return_eval=False),
        policy=actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=frames_remaining,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.optim.mini_batch_size,
    )

    adv_module = GAE(
        gamma=cfg.optim.gamma,
        lmbda=cfg.optim.gae_lambda,
        value_network=critic,
        average_gae=False,
    )

    # Create logger
    exp_name = generate_exp_name("OPUS", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="opus_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode,
                          "project": cfg.logger.project,},
        )

    num_network_updates = 0

    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, ncols=0)
    pbar.update(collected_frames)
    sampling_start = time.time()
    num_mini_batches = cfg.collector.frames_per_batch // cfg.optim.mini_batch_size
    total_network_updates = (
        (frames_remaining // cfg.collector.frames_per_batch)
        * cfg.optim.ppo_epochs
        * num_mini_batches
    )

    #extract cfg variables
    cfg_loss_ppo_epochs = cfg.optim.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.optim.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.optim.clip_epsilon
    cfg_logger_test_interval = cfg.logger.test_interval
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes

    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])


    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
        pbar.update(frames_in_batch)

        # Get training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                    / len(episode_length),
                }
            )

        training_start = time.time()
        for j in range(cfg_loss_ppo_epochs):
            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)
            
            # Update the data buffer
            data_buffer.extend(data_reshape)
            for k, batch in enumerate(data_buffer):
                # Get a data batch
                batch = batch.to(device)

                #TODO: Add annealing
                num_network_updates += 1
                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                critic_loss = loss["loss_critic"]
                actor_loss = loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                actor_loss.backward()
                critic_loss.backward()

                # Update the networks
                actor_optim.step()
                critic_optim.step()
                actor_optim.zero_grad()
                critic_optim.zero_grad()
        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        alpha = 1.0
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon
                if cfg_loss_anneal_clip_eps
                else cfg_loss_clip_epsilon,
            }
        )
        
         # Get test rewards
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if ((i - 1) * frames_in_batch) // cfg_logger_test_interval < (
                i * frames_in_batch
            ) // cfg_logger_test_interval:
                actor.eval()
                eval_start = time.time()
                test_rewards, test_lengths = eval_model(
                    actor, eval_env, num_episodes=cfg_logger_num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_rewards.mean(),
                        "eval/episode_length": test_lengths.mean(),
                        "eval/time": eval_time,
                    }
                )
                actor.train()
                savestate = {
                        'model_actor': actor.state_dict(),
                        'model_critic': critic.state_dict(),
                        'actor_optimizer': actor_optim.state_dict(),
                        'critic_optimizer': critic_optim.state_dict(),
                        "collected_frames": {"collected_frames": collected_frames}
                }
                torch.save(savestate, f"training_snapshot_{collected_frames}.pt")
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)
    
        collector.update_policy_weights_()
        sampling_start = time.time()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()