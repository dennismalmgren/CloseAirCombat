from __future__ import annotations

import contextlib

import math
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch, ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.utils import NestedKey
from torch import distributions as d
import torch.nn.functional as F

from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_ERROR,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)

from torchrl.objectives.common import LossModule
from torchrl.objectives.value import GAE, TD0Estimator, TD1Estimator, TDLambdaEstimator, VTrace
from torchrl.objectives.ppo import PPOLoss


class P3OLoss(PPOLoss):
    """Clipped PPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor_network (ProbabilisticTensorDictSequential): policy operator.
        critic_network (ValueOperator): value operator.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coef (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        critic_coef (scalar, optional): critic loss multiplier when computing the total
            loss. Defaults to ``1.0``.
        loss_critic_type (str, optional): loss function for the value discrepancy.
            Can be one of "l1", "l2" or "smooth_l1". Defaults to ``"smooth_l1"``.
        normalize_advantage (bool, optional): if ``True``, the advantage will be normalized
            before being used. Defaults to ``False``.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        value_target_key (str, optional): [Deprecated, use set_keys(value_target_key=value_target_key) instead]
            The input tensordict key where the target state
            value is expected to be written. Defaults to ``"value_target"``.
        value_key (str, optional): [Deprecated, use set_keys(value_key) instead]
            The input tensordict key where the state
            value is expected to be written. Defaults to ``"state_value"``.
        functional (bool, optional): whether modules should be functionalized.
            Functionalizing permits features like meta-RL, but makes it
            impossible to use distributed models (DDP, FSDP, ...) and comes
            with a little cost. Defaults to ``True``.

    .. note:
      The advantage (typically GAE) can be computed by the loss function or
      in the training loop. The latter option is usually preferred, but this is
      up to the user to choose which option is to be preferred.
      If the advantage key (``"advantage`` by default) is not present in the
      input tensordict, the advantage will be computed by the :meth:`~.forward`
      method.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> advantage = GAE(critic)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)
        >>> # equivalent
        >>> advantage(data)
        >>> losses = ppo_loss(data)

      A custom advantage module can be built using :meth:`~.make_value_estimator`.
      The default is :class:`~torchrl.objectives.value.GAE` with hyperparameters
      dictated by :func:`~torchrl.objectives.utils.default_value_kwargs`.

        >>> ppo_loss = ClipPPOLoss(actor, critic)
        >>> ppo_loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> data = next(datacollector)
        >>> losses = ppo_loss(data)

    .. note::
      If the actor and the value function share parameters, one can avoid
      calling the common module multiple times by passing only the head of the
      value network to the PPO loss module:

        >>> common = SomeModule(in_keys=["observation"], out_keys=["hidden"])
        >>> actor_head = SomeActor(in_keys=["hidden"])
        >>> value_head = SomeValue(in_keys=["hidden"])
        >>> # first option, with 2 calls on the common module
        >>> model = ActorCriticOperator(common, actor_head, value_head)
        >>> loss_module = PPOLoss(model.get_policy_operator(), model.get_value_operator())
        >>> # second option, with a single call to the common module
        >>> loss_module = PPOLoss(ProbabilisticTensorDictSequential(model, actor_head), value_head)

      This will work regardless of whether separate_losses is activated or not.

    """

    def __init__(
        self,
        actor_network: ProbabilisticTensorDictSequential | None = None,
        critic_network: TensorDictModule | None = None,
        *,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coef: float = 0.01,
        critic_coef: float = 1.0,
        loss_critic_type: str = "smooth_l1",
        normalize_advantage: bool = True,
        gamma: float = None,
        separate_losses: bool = False,
        beta: float = 1.0,
        **kwargs,
    ):
        super(P3OLoss, self).__init__(
            actor_network,
            critic_network,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coef=entropy_coef,
            critic_coef=critic_coef,
            loss_critic_type=loss_critic_type,
            normalize_advantage=normalize_advantage,
            gamma=gamma,
            separate_losses=separate_losses,
            **kwargs,
        )
        self.register_buffer("beta", torch.tensor(beta))

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss_objective", "kl"]
            if self.entropy_bonus:
                keys.extend(["entropy", "loss_entropy"])
            if self.loss_critic:
                keys.append("loss_critic")
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values


    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict_copy = tensordict.clone(False)
        try:
            previous_dist = self.actor_network.build_dist_from_params(tensordict)
        except KeyError:
            raise KeyError(
                "The parameters of the distribution were not found. "
                f"Make sure they are provided to {type(self).__name__}"
            )
        advantage = tensordict_copy.get(self.tensor_keys.advantage, None)
        if advantage is None:
            self.value_estimator(
                tensordict_copy,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict_copy.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-6)
            advantage = (advantage - loc) / scale
        log_weight, dist = self._log_weight(tensordict_copy)

        log_weight_minus_1 = log_weight.exp() - 1
        tau = 4.0
        neg_loss = torch.sigmoid(tau * log_weight_minus_1) * 4 / tau * advantage
        with self.actor_network_params.to_module(
            self.actor_network
        ) if self.functional else contextlib.nullcontext():
            current_dist = self.actor_network.get_dist(tensordict_copy)
        try:
            kl = torch.distributions.kl.kl_divergence(previous_dist, current_dist)
        except NotImplementedError:
            x = previous_dist.sample((self.samples_mc_kl,))
            kl = (previous_dist.log_prob(x) - current_dist.log_prob(x)).mean(0)
        kl = kl.unsqueeze(-1)
        neg_loss = neg_loss - self.beta * kl

        td_out = TensorDict({"loss_objective": -neg_loss.mean(),
                             "kl": kl.detach().mean()}, [])

        #if self.entropy_bonus:
        entropy = self.get_entropy_bonus(dist)
        td_out.set("entropy", entropy.mean().detach())  # for logging
        td_out.set("loss_entropy", -self.entropy_coef * entropy.mean()) 

        if self.critic_coef:
            loss_critic = self.loss_critic(tensordict)[0]
            td_out.set("loss_critic", loss_critic.mean())
        return td_out