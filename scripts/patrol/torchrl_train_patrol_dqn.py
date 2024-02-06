import sys
import os
import numpy as np
import torch
import hydra
import logging
import tqdm
import time

from torchrl.envs import (
    GymWrapper, TransformedEnv, RewardSum, 
    StepCounter, Compose, default_info_dict_reader, 
    RewardScaling, step_mdp, ActionMask
)
from torchrl.collectors import RandomPolicy, SyncDataCollector
from tensordict.nn import TensorDictSequential
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl_utils import make_dqn_model, eval_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from torchrl.modules import EGreedyModule
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import BinaryDiscreteTensorSpec
from tensordict import TensorDict

from envs.grid.patrol_env import PatrolEnv




@hydra.main(config_path=".", config_name="config_dqn", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    device = "cpu" if not torch.cuda.device_count() else "cuda"
    #device = "cpu"
    total_frames = 1_000_000
    frames_per_batch = 16
    init_random_frames = 10000
    test_interval = 100_000
    def make_env():
        env = PatrolEnv()
        env = GymWrapper(env, categorical_action_encoding=True)
        env.set_info_dict_reader(default_info_dict_reader(
            keys=["action_mask"],
            spec=[BinaryDiscreteTensorSpec(n=env.action_spec.space.n, device="cpu", dtype=torch.bool)]
                        ))
        env = TransformedEnv(env,
                            Compose(
                                RewardSum(),
                                StepCounter(max_steps=1000),
                                RewardScaling(loc=0, scale=0.01),
                                ActionMask()
                            ))
        return env
    env = make_env()
   
    model = make_dqn_model(env)
    model_eval = make_dqn_model(env)

    greedy_module = EGreedyModule(
        annealing_num_steps=250_000,
        action_mask_key="action_mask",
        eps_init=1.0,
        eps_end=0.05,
        spec=model.spec,
    )
    model_explore = TensorDictSequential(
        model,
        greedy_module,
    ).to(device)

    collector = SyncDataCollector(env, 
                                  policy=model_explore, 
                                  frames_per_batch=frames_per_batch,
                                  total_frames=total_frames,
                                  device="cpu",
                                  storing_device="cpu",
                                  max_frames_per_traj=-1,
                                  init_random_frames=init_random_frames
                                  )
    
        # Create the replay buffer
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=10,
        storage=LazyTensorStorage(
            max_size=1e6,
            device="cpu",
        ),
        batch_size=128,
    )
    
    loss_module = DQNLoss(
        value_network = model,
        loss_function="l2",
        delay_value=True
    )

    loss_module.make_value_estimator(gamma=0.99)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval= 10_000
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)
    
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("DQN", f"Patrol_env")
        logger = get_logger(
            cfg.logger.backend, logger_name="dqn", experiment_name=exp_name
        )

    test_env = make_env()
    # Main loop
    collected_frames = 0
    start_time = time.time()
    sampling_start = time.time()
    num_updates = 1
    max_grad = 10
    num_test_episodes = 10
    q_losses = torch.zeros(num_updates, device=device)
    pbar = tqdm.tqdm(total=total_frames)

    for data in collector:
        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel() 
        collected_frames += current_frames
        greedy_module.step(current_frames)
        replay_buffer.extend(data)

        #get and log training rewards and episode lengths
        episode_rewards = data["next", "episode_reward"][data['next', 'done']]
        if len(episode_rewards) > 0:
            episode_reward_mean = episode_rewards.mean().item()
            episode_length = data['next', 'step_count'][data['next', 'done']]
            episode_length_mean = episode_length.sum().item() / len(episode_length)
            log_info.update(
                {
                    "train/episode_reward": episode_reward_mean,
                    "train/episode_length": episode_length_mean
                }
            )

        if collected_frames < init_random_frames:
            if logger:
                for key, value in log_info.items():
                    logger.log_scalar(key, value, step=collected_frames)
            continue
        
        # optimization steps
        training_start = time.time()

        for j in range(num_updates):
            sampled_tensordict = replay_buffer.sample()
            sampled_tensordict = sampled_tensordict.to(device)

            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad)
            optimizer.step()
            target_net_updater.step()
            q_losses[j].copy_(q_loss.detach())
        training_time = time.time() - training_start
        #action_data = data['action'].reshape(-1, 1)
        #selected_action_qvalues = torch.gather(data['action_value'], 1, data['action'].unsqueeze(1)).squeeze(1)

        log_info.update(
            {
                "train/q_values": (data['chosen_action_value']).sum().item()
                / frames_per_batch,
                "train/q_loss": q_losses.mean().item(),
                "train/epsilon": greedy_module.eps,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_per_batch) // test_interval < (
                collected_frames // test_interval
            ):
                model_weights = TensorDict.from_module(model, as_module=True)
                model_eval_weights = TensorDict.from_module(model_eval, as_module=True)
                model_eval_weights.data.copy_(model_weights.data)
                model_eval.eval()
                eval_start = time.time()
                test_rewards = eval_model(
                    model_eval, test_env, num_episodes=num_test_episodes
                )
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": test_rewards,
                        "eval/eval_time": eval_time,
                    }
                )
                

        # Log all the information
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Training took {execution_time:.2f} seconds to finish")

# #    episode = []
# #    td = env.reset()
# #    done = torch.any(td['done'])
#     while not done:
#         policy(td)
#         td = env.step(td)
#         episode.append(td)
#         td = step_mdp(td)
#         done = torch.any(td['done'])
#     episode_td = torch.stack(episode).to_tensordict()
    

if __name__ == "__main__":
    main()