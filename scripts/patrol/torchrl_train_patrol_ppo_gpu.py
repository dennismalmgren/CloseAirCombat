import sys
import os
import numpy as np
import torch
import hydra
import logging
import tqdm
import time
import copy

from torchrl.envs import (
    GymWrapper, TransformedEnv, RewardSum, 
    StepCounter, Compose, default_info_dict_reader, 
    RewardScaling, step_mdp, ActionMask
)

from torchrl.collectors import RandomPolicy, SyncDataCollector
from tensordict.nn import TensorDictSequential
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from torchrl.modules import EGreedyModule
from torchrl.data import BinaryDiscreteTensorSpec
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE

from envs.grid.patrol_env import PatrolEnv
from envs.grid.patrol_env_gpu import PatrolEnvGpu


from utils_ppo import make_ppo_models, make_parallel_env, eval_model

@hydra.main(config_path=".", config_name="config_ppo", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"

    frame_skip = 1
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    mini_batch_size = cfg.loss.mini_batch_size // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip
    
    # Create models (check utils_atari.py)
    actor, critic = make_ppo_models()
    actor, critic = actor.to(device), critic.to(device)
    
    eval_actor = copy.deepcopy(actor)
    eval_actor = eval_actor.to(device)
    # Create collector
    collector = SyncDataCollector(
        create_env_fn=make_parallel_env(cfg.env.num_envs, device),
        policy=actor,
        
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )

    # Create loss and adv modules
    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        loss_critic_type=cfg.loss.loss_critic_type,
        entropy_coef=cfg.loss.entropy_coef,
        critic_coef=cfg.loss.critic_coef,
        normalize_advantage=True,
    )

    # Create optimizer
    # optim = torch.optim.Adam(
    #     loss_module.parameters(),
    #     lr=cfg.optim.lr,
    #     weight_decay=cfg.optim.weight_decay,
    #     eps=cfg.optim.eps,
    # ) 

    optim = torch.optim.AdamW(
        loss_module.parameters(),
        lr=cfg.optim.lr,
       # weight_decay=cfg.optim.weight_decay,
       # eps=cfg.optim.eps,
    ) 

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("PPO", f"{cfg.logger.exp_name}_{cfg.env.env_name}")
        logger = get_logger(
            cfg.logger.backend,
            logger_name="ppo",
            experiment_name=exp_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
                "mode": cfg.logger.mode,
            },
        )

    # Create test environment
    test_env = make_parallel_env(1, device="cuda", is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (
        (total_frames // frames_per_batch) * cfg.loss.ppo_epochs * num_mini_batches
    )


    # extract cfg variables
    cfg_loss_ppo_epochs = cfg.loss.ppo_epochs
    cfg_optim_anneal_lr = cfg.optim.anneal_lr
    cfg_optim_lr = cfg.optim.lr
    cfg_loss_anneal_clip_eps = cfg.loss.anneal_clip_epsilon
    cfg_loss_clip_epsilon = cfg.loss.clip_epsilon
    cfg_logger_num_test_episodes = cfg.logger.num_test_episodes
    cfg_optim_max_grad_norm = cfg.optim.max_grad_norm
    cfg.loss.clip_epsilon = cfg_loss_clip_epsilon
    cfg_pretrain_save_interval = cfg.model_save.save_interval
    cfg_model_save_experiment = cfg.model_save.experiment_name
    model_name = f"{exp_name}"
    
    load_model = False
    run_as_debug = False
    load_from_debug = False
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    if load_model:
        if run_as_debug:
            if load_from_debug:
                outputs_folder = "../../"
            else:
                outputs_folder = "../../../scripts/patrol/outputs/"
        else:
            if load_from_debug:
                outputs_folder = "../../../../../outputs"
            else:
                outputs_folder = "../../"
        
        run_id = "2024-02-07/07-11-16/"
        model_load_filename = "PPO_PPO_PatrolEnvGrid_10e424dc_24_02_07-07_11_44_iter_9999.pt"
        load_model_dir = outputs_folder + run_id + "saved_models/e2/"
        print('Loading model from ' + load_model_dir)
        loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
        actor_state = loaded_state['actor']
        critic_state = loaded_state['critic']
        optim_state = loaded_state['optimizer']
        actor.load_state_dict(actor_state)
        critic.load_state_dict(critic_state)
        optim.load_state_dict(optim_state)

    losses = TensorDict({}, batch_size=[cfg_loss_ppo_epochs, num_mini_batches])

    sampling_start = time.time()

    for i, data in enumerate(collector):
        log_info = {}
        sampling_time = time.time() - sampling_start
        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

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
                data = adv_module(data.to(device, non_blocking=True))
            data_reshape = data.reshape(-1)
            # Update the data buffer
            data_buffer.extend(data_reshape)

            for k, batch in enumerate(data_buffer):
                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg_optim_anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optim.param_groups:
                        group["lr"] = cfg_optim_lr * alpha
                if cfg_loss_anneal_clip_eps:
                    loss_module.clip_epsilon.copy_(cfg_loss_clip_epsilon * alpha)
                num_network_updates += 1
                # Get a data batch
                batch = batch.to(device, non_blocking=True)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective"
                ).detach()
                loss_sum = (
                    loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )
                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=cfg_optim_max_grad_norm
                )

                # Update the networks
                optim.step()
                optim.zero_grad()

        # Get training losses and times
        training_time = time.time() - training_start
        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/lr": alpha * cfg_optim_lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/clip_epsilon": alpha * cfg_loss_clip_epsilon,
            }
        )

        # Get test rewards
        if ((i - 1) * frames_in_batch * frame_skip) // test_interval < (
            i * frames_in_batch * frame_skip
        ) // test_interval:
            eval_actor.eval()
            actor_weights = TensorDict.from_module(actor, as_module=True)
            eval_actor_weights = TensorDict.from_module(eval_actor, as_module=True)
            eval_actor_weights.data.copy_(actor_weights.data)

            eval_start = time.time()
            test_rewards = eval_model(
                eval_actor, test_env, num_episodes=cfg_logger_num_test_episodes
            )
            eval_time = time.time() - eval_start
            log_info.update(
                {
                    "eval/reward": test_rewards.mean(),
                    "eval/time": eval_time,
                }
            ) 

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        collector.update_policy_weights_()
        sampling_start = time.time()
        if (i * frames_in_batch * frame_skip) // cfg_pretrain_save_interval < (
                (i + 1) * frames_in_batch * frame_skip
            ) // cfg_pretrain_save_interval:
            savestate = {
                        'actor': actor.state_dict(),
                        'critic': critic.state_dict(),
                        'optimizer': optim.state_dict(),
            }
          
            save_model_dir = "saved_models/" + cfg_model_save_experiment
            os.makedirs(save_model_dir, exist_ok=True)

            torch.save(savestate, save_model_dir + f"/{model_name}_iter_{i}.pt")

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()