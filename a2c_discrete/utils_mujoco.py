# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch.nn
import torch.optim

from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.data import CompositeSpec
from torchrl.envs import (
    ClipTransform,
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    StepCounter,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.record import VideoRecorder
from .dropout_modules import ConsistentDropout, ConsistentDropoutModule
from torch.distributions import Categorical
from .continuous_categorical import ContinuousCategorical
# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(
    env_name="HalfCheetah-v4", device="cpu", from_pixels=False, pixels_only=False
):
    env = GymEnv(
        env_name, device=device, from_pixels=from_pixels, pixels_only=pixels_only, categorical_action_encoding=True
    )
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(VecNorm(in_keys=["observation"]))
    env.append_transform(ClipTransform(in_keys=["observation"], low=-10, high=10))
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_ppo_models_state(proof_environment, cfg):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = ContinuousCategorical
    # distribution_kwargs = {
    #     "min": proof_environment.action_spec.space.low,
    #     "max": proof_environment.action_spec.space.high,
    #     "tanh_loc": False,
    # }
    nbins = 101
    supports = [torch.linspace(proof_environment.action_spec.space.low[i], proof_environment.action_spec.space.high[i], nbins) for i in range(num_outputs)]
    support = torch.stack(supports, dim=0)
    distribution_kwargs = {
        "continuous_support": support
        }
    
    # Define policy architecture
    policy_mlp_1 = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs * nbins,  # predict only loc
        num_cells=cfg.network.policy_hidden_sizes,
    )

    # Initialize policy weights
    for layer in policy_mlp_1.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

   # policy_module_2 = ConsistentDropout()
    # Add state-independent normal scale
    policy_mlp = torch.nn.Sequential(
        policy_mlp_1,
    #    policy_module_2,
    #    AddStateIndependentNormalScale(proof_environment.action_spec.shape[-1]),
    )
    return_log_prob = True #if cfg.loss.loss_policy_type == "l2" else False
    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["logits"],
        ),
        in_keys=["logits"],
        out_keys=["discrete_action", "action"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define value architecture
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
    )

    # Initialize value weights
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    # Define value module
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_ppo_models(env_name, cfg):
    proof_environment = make_env(env_name, device="cpu")
    actor, critic = make_ppo_models_state(proof_environment, cfg)
    return actor, critic


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards = np.append(test_rewards, reward.cpu().numpy())
        test_env.apply(dump_video)
    del td_test
    return test_rewards.mean()