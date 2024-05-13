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

import hydra

import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
from .utils import (
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    device = torch.device(cfg.network.device)

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)

    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create off-policy collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device="cpu",
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    # Main loop
    start_time = time.time()
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    save_iter = cfg.logger.save_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps
    max_grad_norm = cfg.optim.max_grad_norm

    sampling_start = time.time()
    for i, tensordict in enumerate(collector):
        sampling_time = time.time() - sampling_start

        # Update weights of the inference policy
        collector.update_policy_weights_()

        pbar.update(tensordict.numel())

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # Optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            losses = TensorDict({}, batch_size=[num_updates])
            norms = TensorDict({}, batch_size=[num_updates])
            q_preds = TensorDict({}, batch_size=[num_updates])
            v_preds = TensorDict({}, batch_size=[num_updates])
            for i in range(num_updates):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                # Compute loss
                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]               
                alpha_loss = loss_td["loss_alpha"]

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    actor_params, max_grad_norm
                )                
                optimizer_actor.step()

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    critic_params, max_grad_norm
                )                
                optimizer_critic.step()

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                alpha_norm = torch.nn.utils.clip_grad_norm_(
                    loss_module.log_alpha, max_grad_norm)                
                optimizer_alpha.step()

                losses[i] = loss_td.select(
                    "loss_actor", "loss_qvalue", "loss_alpha"
                ).detach()
                norms[i] = TensorDict({
                    "actor_norm": actor_norm,
                    "critic_norm": critic_norm,
                    "alpha_norm": alpha_norm
                })
                q_preds[i] = loss_td.select(
                    "q_pred",
                )
                v_preds[i] = loss_td.select(
                    "v_pred",
                )
                # Update qnet_target params
                target_net_updater.step()

                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        training_time = time.time() - training_start
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor").mean().item()
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            metrics_to_log["train/q_norm"] = norms.get("critic_norm").mean().item()
            metrics_to_log["train/actor_norm"] = norms.get("actor_norm").mean().item()
            metrics_to_log["train/alpha_norm"] = norms.get("alpha_norm").mean().item()
            metrics_to_log["train/q_pred"] = q_preds.get("q_pred").mean().item()
            metrics_to_log["train/v_pred"] = v_preds.get("v_pred").mean().item()
            metrics_to_log["train/alpha"] = loss_td["alpha"].item()
            metrics_to_log["train/entropy"] = loss_td["entropy"].item()
            metrics_to_log["train/sampling_time"] = sampling_time
            metrics_to_log["train/training_time"] = training_time
        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_episode_end = (
                    eval_rollout["next", "done"]
                    if eval_rollout["next", "done"].any()
                    else eval_rollout["next", "truncated"]
                )
                eval_time = time.time() - eval_start
                next_tensordict = eval_rollout["next"]
                reward_keys = ["reward"]
                log_rewards = dict()                
                for reward_key in reward_keys:
                    log_rewards[reward_key] = next_tensordict[reward_key].squeeze(-1)

                for key, val in log_rewards.items():
                  #  metrics_to_log["eval/" + key] = val.mean().item()
                    the_reward = val
                    the_return = torch.zeros_like(the_reward)
                    the_return[..., -1] = the_reward[..., -1]
                    for i in range(the_return.shape[-1] - 1, 1, -1):
                        the_return[..., i - 1] = the_reward[..., i - 1] + cfg.optim.gamma * the_return[..., i]
                    if key == "reward":
                        real_return = the_return
                    metrics_to_log["eval/max_return_" + key] = torch.max(the_return)
                    metrics_to_log["eval/mean_return_" + key] = torch.mean(the_return)
                    metrics_to_log["eval/min_return_" + key] = torch.min(the_return)
                    
                metrics_to_log["eval/time"] = eval_time
                metrics_to_log["eval/reward"] = next_tensordict["episode_reward"][eval_episode_end].mean().item()
                eval_rollout_clone = eval_rollout.clone(False).to(device)
                q_pred = loss_module._vmap_qnetworkN0(
                    eval_rollout_clone,
                    loss_module.qvalue_network_params,
                )
                v_pred = loss_module._value_pred(q_pred).min(0)[0].to('cpu')
                q_pred = q_pred['state_action_value'].squeeze(-1)
                q_pred = q_pred.min(0)[0].to('cpu')
                v_pred_diff = real_return - v_pred
                eval_episode_length = eval_rollout["next", "step_count"][eval_episode_end]
                metrics_to_log["eval/v_pred"] = v_pred.mean()
                metrics_to_log["eval/q_pred"] = q_pred.mean()
                metrics_to_log["eval/v_pred_diff"] = v_pred_diff.mean()
                metrics_to_log["eval/time"] = eval_time
                metrics_to_log["eval/episode_length"] = eval_episode_length
                
        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)
        if abs(collected_frames % save_iter) < frames_per_batch:
            savestate = {
                    'model': model.state_dict(),
                    'loss_module': loss_module.state_dict(),
                    'optimizer_actor': optimizer_actor.state_dict(),
                    'optimizer_critic': optimizer_critic.state_dict(),
                    'optimizer_alpha': optimizer_alpha.state_dict(),
                    "collected_frames": {"collected_frames": collected_frames}
            }
            torch.save(savestate, f"training_snapshot_{collected_frames}.pt")
            replay_buffer.dumps(f"replay_buffer_{collected_frames}.rb")

        
        sampling_start = time.time()

    collector.shutdown()
    savestate = {
                    'model': model.state_dict(),
                    'loss_module': loss_module.state_dict(),
                    'optimizer_actor': optimizer_actor.state_dict(),
                    'optimizer_critic': optimizer_critic.state_dict(),
                    'optimizer_alpha': optimizer_alpha.state_dict(),
                    "collected_frames": {"collected_frames": collected_frames}
            }
    torch.save(savestate, f"training_snapshot_{collected_frames}.pt")
    replay_buffer.dumps(f"replay_buffer_{collected_frames}.rb")

    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()