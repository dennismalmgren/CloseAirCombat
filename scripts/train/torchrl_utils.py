# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatTensors,
    Compose,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.data import CompositeSpec

from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.JSBSim.torchrl.jsbsim_wrapper import JSBSimWrapper
from envs.JSBSim.envs.singlecontrol_env_cont import SingleControlEnv
from envs.JSBSim.envs.singlecontrol_env_cont_missile import SingleControlMissileEnv

# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, lowlevelpolicy):
    env = SingleControlMissileEnv(cfg.env_name, lowlevelpolicy)
    wrapped_env = JSBSimWrapper(env, categorical_action_encoding=False)
    return wrapped_env 

def env_maker_ctrl(cfg):
    env = SingleControlEnv(cfg.env_name)
    wrapped_env = JSBSimWrapper(env, categorical_action_encoding=False)
    return wrapped_env 

def apply_env_transforms(env):# max_episode_steps=1000):
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return transformed_env

def make_ppo_environment(cfg, lowlevelpolicy):
    env = env_maker(cfg, lowlevelpolicy=lowlevelpolicy)
    train_env = apply_env_transforms(env)
    eval_env = TransformedEnv(
        env,
        train_env.transform.clone(),
    )
    eval_env.eval()
    return train_env, eval_env

def make_environment(cfg):
    """Make environments for training and evaluation."""

    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: env_maker_ctrl(cfg)),
    )
    parallel_env.set_seed(cfg.seed)

    train_env = apply_env_transforms(parallel_env)#, cfg.collector.max_frames_per_traj)

    eval_env = TransformedEnv(
        ParallelEnv(
            cfg.collector.env_per_collector,
            EnvCreator(lambda cfg=cfg: env_maker_ctrl(cfg)),
        ),
        train_env.transform.clone(),
    )
    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------

def eval_ppo_model(policy, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=policy,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward.cpu())
    del td_test
    return torch.cat(test_rewards, 0).mean()

def make_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_ppo_collector(cfg, train_env, actor_model_explore):
    """Make collector."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.collector.device,
        max_frames_per_traj=-1
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=1000000,
    buffer_scratch_dir=None,
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
                scratch_dir=buffer_scratch_dir,
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
                scratch_dir=buffer_scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----
def make_ppo_modules(cfg, eval_env):
    #begin by using separate nets for value and policy. easier.
    in_keys = ["observation"]
    action_spec = eval_env.action_spec
    if eval_env.batch_size:
        action_spec = action_spec[(0,) * len(eval_env.batch_size)]

    num_outputs = eval_env.action_spec.space.n
    distribution_class = OneHotCategorical
    distribution_kwargs = {}
    policy_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": num_outputs,
        "activation_class": get_activation(cfg),
    }   

    policy_net = MLP(**policy_net_kwargs)

    policy_module = TensorDictModule(
        module = policy_net,
        in_keys=in_keys,
        out_keys=["logits"]
    )

    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=CompositeSpec(action=eval_env.action_spec),
        distribution_class = distribution_class,
        distribution_kwargs = distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=InteractionType.RANDOM,
    )

    value_net = MLP(
        activation_class=torch.nn.ReLU,
        out_features = 1,
        num_cells = cfg.network.hidden_sizes,
    )

    value_module = ValueOperator(
        module = value_net,
        in_keys=in_keys,
    )

    return policy_module, value_module

def make_ppo_models(cfg, eval_env):
    policy_module, value_module = make_ppo_modules(
        cfg, eval_env
    )
    
    with torch.no_grad():
        td = eval_env.reset()
        td = policy_module(td)
        td = value_module(td)

    return policy_module, value_module

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
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(
        **qvalue_net_kwargs,
    )

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.reset()
        td = td.to(device)
        for net in model:
            net(td)
    del td
    eval_env.close()

    return model, model[0]


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    # Create SAC loss
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
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
    else:
        raise NotImplementedError