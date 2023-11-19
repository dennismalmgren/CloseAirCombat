import math
import warnings
from dataclasses import dataclass
from functools import wraps
from numbers import Number
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from tensordict.nn import dispatch, make_functional, TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torch import Tensor
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.data.utils import _find_action_space
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor
from torchrl.modules.tensordict_module.actors import ActorCriticWrapper
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _cache_values,
    _GAMMA_LMBDA_DEPREC_WARNING,
    default_value_kwargs,
    distance_loss,
    ValueEstimators,
)
from torchrl.objectives.value import TD0Estimator, TD1Estimator, TDLambdaEstimator

try:
    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap

    _has_functorch = True
    err = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERROR = err


class MultiDiscreteSACLoss(LossModule):
    """Discrete SAC Loss module.

    Args:
        actor_network (ProbabilisticActor): the actor to be trained
        qvalue_network (TensorDictModule): a single Q-value network that will be multiplicated as many times as needed.
        action_space (str or TensorSpec): Action space. Must be one of
            ``"one-hot"``, ``"mult_one_hot"``, ``"binary"`` or ``"categorical"``,
            or an instance of the corresponding specs (:class:`torchrl.data.OneHotDiscreteTensorSpec`,
            :class:`torchrl.data.MultiOneHotDiscreteTensorSpec`,
            :class:`torchrl.data.BinaryDiscreteTensorSpec` or :class:`torchrl.data.DiscreteTensorSpec`).
        num_qvalue_nets (int, optional): Number of Q-value networks to be trained. Default is 10.
        loss_function (str, optional): loss function to be used for the Q-value. Can be one of `"smooth_l1"`, "l2",
            "l1", Default is "smooth_l1".
        alpha_init (float, optional): initial entropy multiplier.
            Default is 1.0.
        min_alpha (float, optional): min value of alpha.
            Default is None (no minimum value).
        max_alpha (float, optional): max value of alpha.
            Default is None (no maximum value).
        fixed_alpha (bool, optional): whether alpha should be trained to match a target entropy. Default is ``False``.
        target_entropy_weight (float, optional): weight for the target entropy term.
        target_entropy (Union[str, Number], optional): Target entropy for the stochastic policy. Default is "auto".
        delay_qvalue (bool, optional): Whether to separate the target Q value networks from the Q value networks used
            for data collection. Default is ``False``.
        priority_key (str, optional): [Deprecated, use .set_keys(priority_key=priority_key) instead]
            Key where to write the priority value for prioritized replay buffers.
            Default is `"td_error"`.
        separate_losses (bool, optional): if ``True``, shared parameters between
            policy and critic will only be trained on the policy loss.
            Defaults to ``False``, ie. gradients are propagated to shared
            parameters for both policy and critic losses.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            priority (NestedKey): The input tensordict key where the target priority is written to.
                Defaults to ``"td_error"``.
            reward (NestedKey): The input tensordict key where the reward is expected.
                Will be used for the underlying value estimator. Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done. Will be used for the underlying value estimator.
                Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated. Will be used for the underlying value estimator.
                Defaults to ``"terminated"``.
        """

        action: NestedKey = "action"
        value: NestedKey = "state_value"
        action_value: NestedKey = "action_value"
        priority: NestedKey = "td_error"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        log_prob: NestedKey = "log_prob"

    default_keys = _AcceptedKeys()
    default_value_estimator = ValueEstimators.TD0
    delay_actor: bool = False
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_alpha",
        "alpha",
        "entropy",
    ]

    def __init__(
        self,
        actor_network: ProbabilisticActor,
        qvalue_network: TensorDictModule,
        *,
        action_space: Union[str, TensorSpec] = None,
        num_qvalue_nets: int = 2,
        loss_function: str = "smooth_l1",
        alpha_init: float = 1.0,
        min_alpha: float = None,
        max_alpha: float = None,
        fixed_alpha: bool = False,
        target_entropy_weight: float = 0.98,
        delay_qvalue: bool = True,
        priority_key: str = None,
        separate_losses: bool = False,
    ):
        self._in_keys = None
        if not _has_functorch:
            raise ImportError("Failed to import functorch.") from FUNCTORCH_ERROR
        super().__init__()
        self._set_deprecated_ctor_keys(priority_key=priority_key)
        self.nvec = action_space.nvec
        self.convert_to_functional(
            actor_network,
            "actor_network",
            create_target_params=self.delay_actor,
            funs_to_decorate=["forward", "get_dist"], 
        )
        if separate_losses:
            # we want to make sure there are no duplicates in the params: the
            # params of critic must be refs to actor if they're shared
            policy_params = list(actor_network.parameters())
        else:
            policy_params = None
        self.delay_qvalue = delay_qvalue
        self.convert_to_functional(
            qvalue_network,
            "qvalue_network",
            num_qvalue_nets,
            create_target_params=self.delay_qvalue,
            compare_against=policy_params,
        )
        self.num_qvalue_nets = num_qvalue_nets
        self.loss_function = loss_function

        try:
            device = next(self.parameters()).device
        except AttributeError:
            device = torch.device("cpu")

        self.register_buffer("alpha_init", torch.tensor(alpha_init, device=device))
        if bool(min_alpha) ^ bool(max_alpha):
            min_alpha = min_alpha if min_alpha else 0.0
            if max_alpha == 0:
                raise ValueError("max_alpha must be either None or greater than 0.")
            max_alpha = max_alpha if max_alpha else 1e9
        if min_alpha:
            self.register_buffer(
                "min_log_alpha", torch.tensor(min_alpha, device=device).log()
            )
        else:
            self.min_log_alpha = None
        if max_alpha:
            self.register_buffer(
                "max_log_alpha", torch.tensor(max_alpha, device=device).log()
            )
        else:
            self.max_log_alpha = None
        self.fixed_alpha = fixed_alpha
        if fixed_alpha:
            self.register_buffer(
                "log_alpha", torch.tensor(math.log(alpha_init), device=device)
            )
        else:
            self.register_parameter(
                "log_alpha",
                torch.nn.Parameter(torch.tensor(math.log(alpha_init), device=device)),
            )

        if action_space is None:
            warnings.warn(
                "action_space was not specified. DiscreteSACLoss will default to 'one-hot'."
                "This behaviour will be deprecated soon and a space will have to be passed."
                "Check the DiscreteSACLoss documentation to see how to pass the action space. "
            )
            action_space = "one-hot"
        self.action_space = "mult_one_hot" #_find_action_space(action_space)
        #Only supporting auto target entropy
        target_entropy = [-float(np.log(1.0 / num_actions) * target_entropy_weight) for num_actions in action_space.nvec]
        self.register_buffer(
            "target_entropy", torch.mean(torch.tensor(target_entropy, device=device))
        )
        self._vmap_qnetworkN0 = vmap(self.qvalue_network, (None, 0))

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(
                value=self._tensor_keys.value,
                reward=self.tensor_keys.reward,
                done=self.tensor_keys.done,
                terminated=self.tensor_keys.terminated,
            )
        self._set_in_keys()

    def _set_in_keys(self):
        keys = [
            self.tensor_keys.action,
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.done),
            ("next", self.tensor_keys.terminated),
            *self.actor_network.in_keys,
            *[("next", key) for key in self.actor_network.in_keys],
            *self.qvalue_network.in_keys,
        ]
        self._in_keys = list(set(keys))

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = None
        if tensordict.ndimension() > 1:
            shape = tensordict.shape
            tensordict_reshape = tensordict.reshape(-1)
        else:
            tensordict_reshape = tensordict

        loss_value, metadata_value = self._value_loss(tensordict_reshape)
        loss_actor, metadata_actor = self._actor_loss(tensordict_reshape)
        loss_alpha = self._alpha_loss(
            log_prob=metadata_actor["log_prob"],
        )

        tensordict_reshape.set(self.tensor_keys.priority, metadata_value["td_error"])
        if loss_actor.shape != loss_value.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {loss_actor.shape}, and {loss_value.shape}"
            )
        if shape:
            tensordict.update(tensordict_reshape.view(shape))
        entropy = -metadata_actor["log_prob"].mean()
        out = {
            "loss_actor": loss_actor.mean(),
            "loss_qvalue": loss_value.mean(),
            "loss_alpha": loss_alpha.mean(),
            "alpha": self._alpha,
            "entropy": entropy,
        }
        return TensorDict(out, [])

    def _compute_target(self, tensordict) -> Tensor:
        r"""Value network for SAC v2.

        SAC v2 is based on a value estimate of the form:

        .. math::

          V = Q(s,a) - \alpha * \log p(a | s)

        This class computes this value given the actor and qvalue network

        """
        tensordict = tensordict.clone(False)
        # get actions and log-probs
        with torch.no_grad():
            next_tensordict = tensordict.get("next").clone(False)

            # get probs and log probs for actions computed from "next"
            next_dist = self.actor_network.get_dist(
                next_tensordict, params=self.actor_network_params
            )
            next_log_probs = []
            next_probs = []
            for key, dist in next_dist.dists.items():
                next_prob = dist.probs
                next_log_prob = torch.log(torch.where(next_prob == 0, 1e-8, next_prob))
                next_log_probs.append(next_log_prob)
                next_probs.append(next_prob)
                #not sure what to do with these just yet.
#            next_prob = next_dist.probs
#            next_log_prob = torch.log(torch.where(next_prob == 0, 1e-8, next_prob))

            # get q-values for all actions
            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            next_action_value = next_tensordict_expand.get(
                self.tensor_keys.action_value
            )
            start = stop = 0
            next_state_values = []
            for act_dim, next_prob, next_log_prob in zip(self.nvec, next_probs, next_log_probs):
                stop += act_dim
                this_next_state_value = next_action_value[..., start:stop].min(0)[0] - self._alpha * next_log_prob
                this_next_state_value = (next_prob * this_next_state_value).sum(-1).unsqueeze(-1)
                next_state_values.append(this_next_state_value)
                start = stop
            next_state_values = torch.cat(next_state_values, dim=-1)
            next_state_value = torch.sum(next_state_values, dim=-1).unsqueeze(-1)
            # like in continuous SAC, we take the minimum of the value ensemble and subtract the entropy term
#            next_state_value = next_action_value.min(0)[0] - self._alpha * next_log_prob
            # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
#            next_state_value = (next_prob * next_state_value).sum(-1).unsqueeze(-1)

            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            target_value = self.value_estimator.value_estimate(tensordict).squeeze(-1)
            return target_value

    def _value_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        target_value = self._compute_target(tensordict)
        tensordict_expand = self._vmap_qnetworkN0(
            tensordict.select(*self.qvalue_network.in_keys),
            self.qvalue_network_params,
        )

        action_value = tensordict_expand.get(self.tensor_keys.action_value)
        action = tensordict.get(self.tensor_keys.action)
        action = action.expand((action_value.shape[0], *action.shape))  # Add vmap dim

        # TODO this block comes from the dqn loss, we need to swap all these with a proper
        #  helper function which selects the value given the action for all discrete spaces
        if self.action_space == "categorical":
            if action.shape != action_value.shape:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            chosen_action_value = torch.gather(action_value, -1, index=action).squeeze(
                -1
            )
        elif self.action_space == "mult_one_hot":
            start = stop = 0
            chosen_action_value = []
            for act_dim in self.nvec:
                stop += act_dim #todo: review ths.
                this_action_value = (action_value[..., start:stop] * action[..., start:stop]).sum(-1)
                chosen_action_value.append(this_action_value)
                start = stop
            chosen_action_value = torch.stack(chosen_action_value, dim=-2)

            chosen_action_value = torch.sum(chosen_action_value, dim=-2).squeeze()
        else:
            action = action.to(torch.float)
            chosen_action_value = (action_value * action).sum(-1)

        td_error = torch.abs(chosen_action_value - target_value)
        loss_qval = distance_loss(
            chosen_action_value,
            target_value.expand_as(chosen_action_value),
            loss_function=self.loss_function,
        ).mean(0)

        metadata = {
            "td_error": td_error.detach().max(0)[0],
        }
        return loss_qval, metadata

    def _actor_loss(
        self, tensordict: TensorDictBase
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # get probs and log probs for actions
        dist = self.actor_network.get_dist(
            tensordict,
            params=self.actor_network_params,
        )
        log_probs = []
        probs = []
        for key, dist in dist.dists.items():
            prob = dist.probs
            log_prob = torch.log(torch.where(prob == 0, 1e-8, prob))
            log_probs.append(log_prob)
            probs.append(prob)
            
        # prob = dist.probs
        # log_prob = torch.log(torch.where(prob == 0, 1e-8, prob))
        #TODO: DOES THIS REALLY WORK FOR ONE-HOT?
        prob = torch.cat(probs, dim=-1)
        log_prob = torch.cat(log_probs, dim=-1)
        td_q = tensordict.select(*self.qvalue_network.in_keys)
        td_q = self._vmap_qnetworkN0(
            td_q, self._cached_detached_qvalue_params  # should we clone?
        )
        min_q = td_q.get(self.tensor_keys.action_value).min(0)[0]

        if log_prob.shape != min_q.shape:
            raise RuntimeError(
                f"Losses shape mismatch: {log_prob.shape} and {min_q.shape}"
            )

        # like in continuous SAC, we take the entropy term and subtract the minimum of the value ensemble
        loss = self._alpha * log_prob - min_q
        # unlike in continuous SAC, we can compute the exact expectation over all discrete actions
        loss = (prob * loss).sum(-1)

        return loss, {"log_prob": (log_prob * prob).sum(-1).detach()}

    def _alpha_loss(self, log_prob: Tensor) -> Tensor:
        if self.target_entropy is not None:
            # we can compute this loss even if log_alpha is not a parameter
            alpha_loss = -self.log_alpha * (log_prob + self.target_entropy)
        else:
            # placeholder
            alpha_loss = torch.zeros_like(log_prob)
        return alpha_loss

    @property
    def _alpha(self):
        if self.min_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            alpha = self.log_alpha.exp()
        return alpha

    @property
    @_cache_values
    def _cached_detached_qvalue_params(self):
        return self.qvalue_network_params.detach()

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        hp.update(hyperparams)
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(
                **hp,
                value_network=None,
            )
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(
                **hp,
                value_network=None,
            )
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} it not implemented for loss {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            self._value_estimator = TDLambdaEstimator(
                **hp,
                value_network=None,
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        tensor_keys = {
            "value": self.tensor_keys.value,
            "value_target": "value_target",
            "reward": self.tensor_keys.reward,
            "done": self.tensor_keys.done,
            "terminated": self.tensor_keys.terminated,
        }
        self._value_estimator.set_keys(**tensor_keys)
