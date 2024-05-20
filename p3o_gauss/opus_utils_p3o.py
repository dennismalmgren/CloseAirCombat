# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import torch

from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim, Tensor
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    TransformedEnv,
    RewardScaling,
    VecNorm,
    ClipTransform
)
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.data import CompositeSpec
from tensordict.nn import AddStateIndependentNormalScale
import torch.nn.functional as F
from torch import nn

from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from tensordict import TensorDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.JSBSim.torchrl.jsbsim_wrapper import JSBSimWrapper
from envs.JSBSim.envs import OpusTrainingEnv

# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg):
    env = OpusTrainingEnv(cfg.env.name)
    wrapped_env = JSBSimWrapper(env, categorical_action_encoding=False)
    return wrapped_env 


def apply_env_transforms(env):# max_episode_steps=1000):
    reward_keys = list(env.reward_spec.keys())
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=1000),
            DoubleToFloat(),
            VecNorm(in_keys=["observation"], decay=0.99999, eps=1e-2),
            ClipTransform(in_keys=["observation"], low=-10, high=10),
            RewardSum(in_keys=reward_keys,
                      reset_keys=reward_keys),
            CatFrames(5, dim=-1, in_keys=['observation'])
        ),
    )
    return transformed_env

def make_ppo_environment(cfg):
    env = env_maker(cfg)
    train_env = apply_env_transforms(env)
    eval_env = TransformedEnv(
        env,
        train_env.transform.clone(),
    )
    eval_env.eval()
    return train_env, eval_env

def make_environment(cfg, return_eval=True):
    """Make environments for training and evaluation."""

    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: env_maker(cfg)),
    )
    parallel_env.set_seed(cfg.env.seed)
    train_env = apply_env_transforms(parallel_env)#, cfg.collector.max_frames_per_traj)
    if not return_eval:
        return train_env

    eval_env = TransformedEnv(
        ParallelEnv(
            1,
            EnvCreator(lambda cfg=cfg: env_maker(cfg)),
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
class SupportOperator(nn.Module):
    def __init__(self, support):
        super().__init__()
        self.register_buffer("support", support)

    def forward(self, x):
        return (x.softmax(-1) * self.support).sum(-1, keepdim=True)

class ClampOperator(nn.Module):
    def __init__(self, vmin, vmax):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    def forward(self, x):
        return torch.clamp(x, self.vmin, self.vmax)

class TSSR(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        abs_input = torch.abs(input)
        condition = abs_input <= 1
        result = torch.where(
            condition,
            input,
            torch.sign(input) * (2 * torch.sqrt(abs_input) - 1),
        )
        return result
# class RandomFourierFeatures(nn.Module):
#     def __init__(self, input_dim, num_features, scale=1.0):
#         super(RandomFourierFeatures, self).__init__()
#         self.register_buffer("num_features", torch.tensor(num_features))
#         self.scale = scale
#         # Generate W and b parameters
#         self.W = nn.Parameter(torch.randn(num_features, input_dim) * scale, requires_grad=False)
#         self.b = nn.Parameter(torch.rand(num_features) * 2 * torch.pi, requires_grad=False)

#     def forward(self, x):
#         # x should be (B, N)
#         # Apply the RFF mapping: sqrt(2/D) * cos(Wx + b)
#         transformed = torch.cos(torch.matmul(x, self.W.T) + self.b)
#         return torch.sqrt(2. / self.num_features) * transformed

# class FourierEmbeddingModule(nn.Module):
#     def __init__(self, input_dim, num_features):
#         super(FourierEmbeddingModule, self).__init__()
#         self.rff = RandomFourierFeatures(input_dim, num_features)

#     def forward(self, observation):
#         # Reshape from (B, N) to (B, 5, N // 5)
#         batch_shape = observation.shape[:-1]
#         N = observation.shape[-1]
#         x_reshaped = observation.view(*batch_shape, 5, N // 5)
        
#         # Embed the first element (segment) of the reshaped tensor
#         first_embedded = self.rff(x_reshaped[..., :, :1])  # Apply RFF to the first segment
#         first_embedded = first_embedded.reshape(*batch_shape, 5, -1)
#         # Concatenate the embedded first segment with the remainder of the observations
#         remaining = x_reshaped[..., 1:]
#         output = torch.cat((first_embedded, remaining), dim=-1)
#         output = output.reshape(*batch_shape, -1)
        
#         return output
    
def make_ppo_models_state(cfg, proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": proof_environment.action_spec.space.low,
        "max": proof_environment.action_spec.space.high,
        "tanh_loc": False,
    }

    #num_fourier_features = 64
    #fourier_embedding = FourierEmbeddingModule(1, num_fourier_features)

    # Define policy architecture
    policy_mlp = MLP(
        in_features=input_shape[-1], #+ num_fourier_features * 5 - 5,
        activation_class=torch.nn.SiLU,
        out_features=num_outputs,  # predict only loc
        num_cells=cfg.network.policy_hidden_sizes,
        norm_class=torch.nn.LayerNorm,
        norm_kwargs=[{"elementwise_affine": False,
                     "normalized_shape": hidden_size} for hidden_size in cfg.network.policy_hidden_sizes],
    )
    
    # Initialize policy weights
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    clamp_operator = ClampOperator(-10, 10)

    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        #fourier_embedding,
        policy_mlp,
        clamp_operator,
        AddStateIndependentNormalScale(
            proof_environment.action_spec.shape[-1], scale_lb=1e-8
        ),
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    nbins = cfg.network.nbins
    Vmin = cfg.network.vmin
    Vmax = cfg.network.vmax

    support = torch.linspace(Vmin, Vmax, nbins)

    value_net_kwargs = {
        "in_features": input_shape[-1],# + num_fourier_features * 5 - 5,
        "activation_class": torch.nn.SiLU,
        "out_features": nbins,
        "num_cells": cfg.network.value_hidden_sizes,
    }

    value_net = MLP(
        **value_net_kwargs,
    )
    last_layer = value_net[-1]

    bias_data = torch.tensor([-0.01] * (40) + [-100.0] * (nbins - (40)))
    last_layer.bias.data = bias_data
    value_net = nn.Sequential(
        #fourier_embedding,
        value_net
    )
    in_keys = ["observation"]
    value_module_1 = TensorDictModule(
        in_keys=in_keys,
        out_keys=["state_value_logits"],
        module=value_net,
    )
    support_network = SupportOperator(support)
    value_module_2 = TensorDictModule(support_network, in_keys=["state_value_logits"], out_keys=["state_value"])
    value_module = TensorDictSequential(value_module_1, value_module_2)

#### OLD VALUE MODULE
    # value_mlp = MLP(
    #     in_features=input_shape[-1],
    #     activation_class=torch.nn.Tanh,
    #     out_features=1,
    #     num_cells=cfg.network.value_hidden_sizes,
    # )

    # # Initialize value weights
    # for layer in value_mlp.modules():
    #     if isinstance(layer, torch.nn.Linear):
    #         torch.nn.init.orthogonal_(layer.weight, 0.01)
    #         layer.bias.data.zero_()

    # # Define value module
    # value_module = ValueOperator(
    #     value_mlp,
    #     in_keys=["observation"],
    # )
    
#### END OLD VALUE MODULE
    return policy_module, value_module, support

# def make_ppo_modules(cfg, proof_environment):
#     input_shape = proof_environment.observation_spec["observation"].shape
#     action_spec = proof_environment.action_spec
#     # Define policy output distribution class
#     #num_outputs = proof_environment.action_spec.shape[-1]
#     distribution_class = TanhNormal
#     distribution_kwargs = {
#         "min": proof_environment.action_spec.space.low,
#         "max": proof_environment.action_spec.space.high,
#         "tanh_loc": False,
#     }

#     actor_net_kwargs = {
#         "in_features":input_shape[-1],
#         "num_cells": cfg.network.policy_hidden_sizes,
#         "out_features": 2 * action_spec.shape[-1],
#         "activation_class": torch.nn.ELU,
#         "activate_last_layer": True
#     }

#     # Define policy architecture
#     actor_net = MLP(**actor_net_kwargs)
#     #actor_next_layer = nn.Linear(256, 2 * action_spec.shape[-1])
#     #actor_next_layer = torch.nn.utils.spectral_norm(actor_next_layer)

#     actor_extractor = NormalParamExtractor(
#         scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
#         scale_lb=cfg.network.scale_lb,
#     )
#     actor_net = nn.Sequential(actor_net, actor_extractor)

#     in_keys = ["observation"]
#     in_keys_actor = in_keys
#     actor_module = TensorDictModule(
#         actor_net,
#         in_keys=in_keys_actor,
#         out_keys=[
#             "loc",
#             "scale",
#         ],
#     )

#     # Add probabilistic sampling of the actions
#     actor = ProbabilisticActor(
#         spec=action_spec,
#         in_keys=["loc", "scale"],
#         module=actor_module,
#         distribution_class=distribution_class,
#         distribution_kwargs=distribution_kwargs, 
#         return_log_prob=True,
#         default_interaction_type=ExplorationType.RANDOM,
#     )
#   # Define Critic Network
    
#     # Define value architecture
#     nbins = cfg.network.nbins

#     value_net_kwargs = {
#         "in_features": input_shape[-1],
#         "activation_class": torch.nn.ReLU,
#         "out_features": nbins,
#         "num_cells": cfg.network.value_hidden_sizes,
#     }

#     value_net = MLP(
#         **value_net_kwargs,
#     )
#     last_layer = value_net[-1]

#     bias_data = torch.tensor([-0.01] * (20) + [-100.0] * (nbins - (20)))
#     last_layer.bias.data = bias_data

#     in_keys = ["observation"]
#     value_module_1 = TensorDictModule(
#         in_keys=in_keys,
#         out_keys=["state_value_logits"],
#         module=value_net,
#     )
#     Vmin = cfg.network.vmin
#     Vmax = cfg.network.vmax

#     support = torch.linspace(Vmin, Vmax, nbins)
#     support_network = SupportOperator(support)
#     value_module_2 = TensorDictModule(support_network, in_keys=["state_value_logits"], out_keys=["state_value"])
#     value_module = TensorDictSequential(value_module_1, value_module_2)
#     return actor, value_module, support

def load_observation_statistics(statistics_file, run_folder_name=""):
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    load_from_saved_models = run_folder_name == ""
    if load_from_saved_models:
        outputs_folder = "../../../saved_models/"
    else:
        outputs_folder = "../../"

    model_load_filename = f"{statistics_file}.td"
    load_model_dir = outputs_folder + run_folder_name
    print('Loading statistics from ' + load_model_dir)
    loaded_state = TensorDict.load_memmap(load_model_dir + f"{model_load_filename}")
    return loaded_state

def load_model_state(model_name, run_folder_name=""):
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    load_from_saved_models = run_folder_name == ""
    if load_from_saved_models:
        outputs_folder = "../../../saved_models/"
    else:
        outputs_folder = "../../"

    model_load_filename = f"{model_name}.pt"
    load_model_dir = outputs_folder + run_folder_name
    print('Loading model from ' + load_model_dir)
    loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
    return loaded_state

def make_agent(cfg, eval_env, device):
    policy_module, value_module, support = make_ppo_models_state(
        cfg, eval_env
    )
    support = support.to(device)
    policy_module = policy_module.to(device)
    value_module = value_module.to(device)

    with torch.no_grad():
        td = eval_env.reset()
        td = td.to(device)
        td = policy_module(td)
        td = value_module(td)
        td = td.to("cpu")

    return policy_module, value_module, support

# def make_agent(cfg, eval_env, device):
#     policy_module, value_module, support = make_ppo_modules(
#         cfg, eval_env
#     )
#     policy_module = policy_module.to(device)
#     value_module = value_module.to(device)
#     support = support.to(device)
#     with torch.no_grad():
#         td = eval_env.reset()
#         td = td.to(device)
#         td = policy_module(td)
#         td = value_module(td)
#         td = td.to("cpu")

#     return policy_module, value_module, support

def eval_model(actor, test_env, reward_keys, num_episodes=3):
    test_rewards = dict()
    test_returns = dict()
    test_lengths = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        
        for key in reward_keys:
            episode_reward = td_test["next", "episode_" + key][td_test["next", "done"]]
            test_rewards["episode_" + key] = test_rewards.get(key, []) + [episode_reward.cpu().item()]
          
        episode_length = td_test["next", "step_count"][td_test["next", "done"]]

    #    test_rewards.append(reward.cpu())
        test_lengths.append(episode_length.type(torch.float32).cpu())
    del td_test
    return test_rewards, torch.cat(test_lengths, 0).mean()

def split_critic_params(critic_params):
    critic1_params = []
    critic2_params = []

    for param in critic_params:
        data1, data2 = param.data.chunk(2, dim=0)
        critic1_params.append(nn.Parameter(data1))
        critic2_params.append(nn.Parameter(data2))
    return critic1_params, critic2_params



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