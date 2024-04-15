# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    Compose,
    DMControlEnv,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
    CatFrames
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, DistributionalQValueActor

from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from sac_gauss_jsbsim.sac_gauss_loss import SACGaussLoss

from envs.JSBSim.torchrl.jsbsim_wrapper import JSBSimWrapper
from envs.JSBSim.envs import OpusTrainingEnv

# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, device="cpu"):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium"):
        with set_gym_backend(lib):
            return GymEnv(
                cfg.env.name,
                device=device,
            )
    elif lib == "dm_control":
        env = DMControlEnv(cfg.env.name, cfg.env.task)
        return TransformedEnv(
            env, CatTensors(in_keys=env.observation_spec.keys(), out_key="observation")
        )
    else:
        env = OpusTrainingEnv(cfg.env.name)
        wrapped_env = JSBSimWrapper(env, categorical_action_encoding=False)
        return wrapped_env 
        

def apply_env_transforms(env, cfg, max_episode_steps=1000):
    lib = cfg.env.library
    if lib in ("gym", "gymnasium") or lib == "dm_control":
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_episode_steps),
                DoubleToFloat(),
                RewardSum(),
            ),
        )
    else:
        reward_keys = ["reward"]
        for reward_function in env.task[0].reward_functions:
                for key in reward_function.reward_item_names:
                    reward_keys.append(key)
        transformed_env = TransformedEnv(
            env,
            Compose(
                InitTracker(),
                StepCounter(max_episode_steps),
                DoubleToFloat(),
                RewardSum(in_keys=reward_keys,
                        reset_keys=reward_keys
                                   ),
                CatFrames(5, dim=-1, in_keys=["observation"])
            ),
        )
    return transformed_env


def make_environment(cfg):
    """Make environments for training and evaluation."""
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: env_maker(cfg)),
        serial_for_single=True,
    )
    parallel_env.set_seed(cfg.env.seed)
    reward_keys = parallel_env.task[0].get_reward_keys()
    train_env = apply_env_transforms(parallel_env, cfg, cfg.env.max_episode_steps)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(lambda cfg=cfg: env_maker(cfg)),
            serial_for_single=True,
        ),
        train_env.transform.clone(),
    )
    return train_env, eval_env, reward_keys


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, collected_frames, frames_remaining):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=max(0, cfg.collector.init_random_frames - collected_frames),
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=frames_remaining,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    # Define Actor Network
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "min": action_spec.space.low,
        "max": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    #lets assume we have 3 outputs.
    
    nbins = 101
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": nbins,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue1 = TensorDictModule(
        in_keys=["action"] + in_keys,
        out_keys=["state_action_value"],
        module=qvalue_net,
    )

    qvalue2 = TensorDictModule(lambda x: x.log_softmax(-1), ["state_action_value"], ["state_action_value"])

    qvalue = TensorDictSequential(qvalue1, qvalue2)

    model = nn.ModuleList([actor, qvalue]).to(device)
    Vmin = -500
    Vmax = 500
    N_bins = nbins - 6
    delta = (Vmax - Vmin) / (N_bins - 1)
    Vmin_final = Vmin - 3 * delta
    Vmax_final = Vmax + 3 * delta

    support = torch.linspace(Vmin_final, Vmax_final, nbins).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0], support


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model, support):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACGaussLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
        support = support,
        gamma = cfg.optim.gamma
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    # Define Target Network Updater
    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    elif cfg.network.activation == "swish":
        return nn.SiLU
    else:
        raise NotImplementedError