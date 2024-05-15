# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import torch
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
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
    RewardScaling
)
from torchrl.envs.utils import step_mdp
from torchrl.envs.libs.gym import GymEnv, set_gym_backend
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, LSTMModule
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torchrl.data import CompositeSpec
from tensordict.nn import AddStateIndependentNormalScale

from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.JSBSim.torchrl.jsbsim_wrapper import JSBSimWrapper
from envs.JSBSim.envs import OpusTrainingEnv

# ====================================================================
# Environment utils
# -----------------


class SequentialSampler(Sampler):
    """A data-consuming sampler that ensures that the same sample is not present in consecutive batches.

    Args:
        drop_last (bool, optional): if ``True``, the last incomplete sample (if any) will be dropped.
            If ``False``, this last sample will be kept and (unlike with torch dataloaders)
            completed with other samples from a fresh indices permutation.
            Defaults to ``False``.
        shuffle (bool, optional): if ``False``, the items are not randomly
            permuted. This enables to iterate over the replay buffer in the
            order the data was collected. Defaults to ``True``.

    *Caution*: If the size of the storage changes in between two calls, the samples will be re-shuffled
    (as we can't generally keep track of which samples have been sampled before and which haven't).

    Similarly, it is expected that the storage content remains the same in between two calls,
    but this is not enforced.

    When the sampler reaches the end of the list of available indices, a new sample order
    will be generated and the resulting indices will be completed with this new draw, which
    can lead to duplicated indices, unless the :obj:`drop_last` argument is set to ``True``.

    """

    def __init__(self, drop_last: bool = False, shuffle: bool = True):
        self._sample_list = None
        self.len_storage = 0
        self.drop_last = drop_last
        self._ran_out = False
        self.shuffle = shuffle

    def dumps(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)

        with open(path / "sampler_metadata.json", "w") as file:
            json.dump(
                self.state_dict(),
                file,
            )

    def loads(self, path):
        with open(path / "sampler_metadata.json", "r") as file:
            metadata = json.load(file)
        self.load_state_dict(metadata)

    def _get_sample_list(self, storage: Storage, len_storage: int):
        if storage is None:
            device = self._sample_list.device
        else:
            device = storage.device if hasattr(storage, "device") else None

        if self.shuffle:
            _sample_list = torch.randperm(len_storage, device=device)
        else:
            _sample_list = torch.arange(len_storage, device=device)
        self._sample_list = _sample_list

    def _single_sample(self, len_storage, batch_size):
        index = self._sample_list[:batch_size]
        self._sample_list = self._sample_list[batch_size:]

        # check if we have enough elements for one more batch, assuming same batch size
        # will be used each time sample is called
        if self._sample_list.shape[0] == 0 or (
            self.drop_last and len(self._sample_list) < batch_size
        ):
            self.ran_out = True
            self._get_sample_list(storage=None, len_storage=len_storage)
        else:
            self.ran_out = False
        return index

    def _storage_len(self, storage):
        return len(storage)

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Any, dict]:
        len_storage = self._storage_len(storage)
        if len_storage == 0:
            raise RuntimeError(_EMPTY_STORAGE_ERROR)
        if not len_storage:
            raise RuntimeError("An empty storage was passed")
        if self.len_storage != len_storage or self._sample_list is None:
            self._get_sample_list(storage, len_storage)
        if len_storage < batch_size and self.drop_last:
            raise ValueError(
                f"The batch size ({batch_size}) is greater than the storage capacity ({len_storage}). "
                "This makes it impossible to return a sample without repeating indices. "
                "Consider changing the sampler class or turn the 'drop_last' argument to False."
            )
        self.len_storage = len_storage
        index = self._single_sample(len_storage, batch_size)
        if storage.ndim > 1:
            index = torch.unravel_index(index, storage.shape)
        # we 'always' return the indices. The 'drop_last' just instructs the
        # sampler to turn to `ran_out = True` whenever the next sample
        # will be too short. This will be read by the replay buffer
        # as a signal for an early break of the __iter__().
        return index, {}

    @property
    def ran_out(self):
        return self._ran_out

    @ran_out.setter
    def ran_out(self, value):
        self._ran_out = value

    def _empty(self):
        self._sample_list = None
        self.len_storage = 0
        self._ran_out = False

    def state_dict(self) -> Dict[str, Any]:
        return OrderedDict(
            len_storage=self.len_storage,
            _sample_list=self._sample_list,
            drop_last=self.drop_last,
            _ran_out=self._ran_out,
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.len_storage = state_dict["len_storage"]
        self._sample_list = state_dict["_sample_list"]
        self.drop_last = state_dict["drop_last"]
        self._ran_out = state_dict["_ran_out"]

    def __repr__(self):
        if self._sample_list is not None:
            perc = len(self._sample_list) / self.len_storage * 100
        else:
            perc = 0.0
        return f"{self.__class__.__name__}({perc: 4.4f}% sampled)"



def apply_env_transforms(env):# max_episode_steps=1000):
    reward_keys = list(env.reward_spec.keys())
    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=1000),
            DoubleToFloat(),
            RewardSum(in_keys=reward_keys,
                      reset_keys=reward_keys),
           # CatFrames(5, dim=-1, in_keys=['observation'])
        ),
    )
    return transformed_env


def make_eval_environment(cfg):
    def env_maker(cfg):
        env = OpusTrainingEnv(cfg.env.name)
        wrapped_env = JSBSimWrapper(env, categorical_action_encoding=False)
        return wrapped_env 
    env = env_maker(cfg)
    eval_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(max_steps=1000),
            DoubleToFloat(),
            RewardSum(in_keys=list(env.reward_spec.keys()),
                      reset_keys=list(env.reward_spec.keys())),
        ),
    )
    eval_env.eval()
    return eval_env

def make_train_environment(cfg, lstm_module):
    """Make environments for training and evaluation."""
    def env_maker(cfg):
        env = OpusTrainingEnv(cfg.env.name)
        wrapped_env = JSBSimWrapper(env, categorical_action_encoding=False)
        wrapped_env = wrapped_env.append_transform(lstm_module.make_tensordict_primer())
        return wrapped_env 
    
    parallel_env = ParallelEnv(
        cfg.collector.env_per_collector,
        EnvCreator(lambda cfg=cfg: env_maker(cfg)),
    )
    parallel_env.set_seed(cfg.env.seed)
    train_env = apply_env_transforms(parallel_env)#, cfg.collector.max_frames_per_traj)
    return train_env


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
def make_ppo_modules(cfg, proof_environment):
    input_shape = proof_environment.observation_spec["observation"].shape
    
    # Define policy output distribution class
    num_outputs = proof_environment.action_spec.shape[-1]
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": proof_environment.action_spec.space.low,
        "max": proof_environment.action_spec.space.high,
        "tanh_loc": False,
    }

    feature_size = 64

    policy_features_network = MLP(
        in_features = input_shape[-1],
        activation_class=torch.nn.ReLU,
        out_features = feature_size
    )

    policy_features_module = TensorDictModule(
        module=policy_features_network,
        in_keys = ["observation"],
        out_keys=["embed"]
    )

    lstm_output_size = 128
    policy_lstm = LSTMModule(
        input_size=feature_size,  
        hidden_size = 128,
        in_key="embed",
        out_key="embed" #? why not embed..
    )
    # for name, param in policy_lstm.named_parameters():
    #         if "bias" in name:
    #             nn.init.constant_(param, 0)
    #         elif "weight" in name:
    #             nn.init.orthogonal_(param, 1.0)

    # Define policy architecture
    policy_action_network = MLP(
        in_features=lstm_output_size,
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,  # predict only loc
        num_cells=cfg.network.hidden_sizes,
    )

    # Initialize policy weights
    for layer in policy_action_network.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_action_network = torch.nn.Sequential(
        policy_action_network,
        AddStateIndependentNormalScale(
            proof_environment.action_spec.shape[-1], scale_lb=1e-8
        ),
    )

    policy_action_module = TensorDictModule(
        module=policy_action_network,
        in_keys=["embed"],
        out_keys=["loc", "scale"],
    )

    # Add state-independent normal scale
    policy_inference_module = TensorDictSequential(
        policy_features_module,
        policy_lstm,
        policy_action_module,
    )

    # Add probabilistic sampling of the actions
    policy_inference_module = ProbabilisticActor(
        policy_inference_module,
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    policy_training_module = TensorDictSequential(
        policy_features_module,
        policy_lstm.set_recurrent_mode(True),
        policy_action_module,
    )

    policy_training_module = ProbabilisticActor(
        policy_training_module,
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
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

    return policy_inference_module, policy_training_module, policy_lstm, value_module

def make_agent(cfg, eval_env, device):
    policy_inference_module, policy_training_module, policy_lstm, value_module = make_ppo_modules(
        cfg, eval_env
    )
    eval_env.append_transform(policy_lstm.make_tensordict_primer())
    policy_inference_module = policy_inference_module.to(device)
    value_module = value_module.to(device)
    with torch.no_grad():
        td = eval_env.reset()
        td = td.to(device)
        td = policy_inference_module(td)
        td = value_module(td)        
    return policy_inference_module, policy_training_module, policy_lstm, value_module

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