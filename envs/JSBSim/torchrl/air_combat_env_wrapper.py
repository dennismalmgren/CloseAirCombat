from __future__ import annotations
import itertools

import importlib.util
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    BoundedTensorSpec, 
    CompositeSpec,
    DEVICE_TYPING,
    DiscreteTensorSpec,
    LazyStackedCompositeSpec,
    MultiDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
import gymnasium as gym

from torchrl.data.utils import numpy_to_torch_dtype_dict
from torchrl.envs.common import _EnvWrapper, EnvBase
from torchrl.envs.libs.gym import gym_backend, set_gym_backend
from torchrl.envs.utils import (
    _classproperty,
    _selective_unsqueeze,
    check_marl_grouping,
    MarlGroupMapType,
)

from envs.JSBSim.envs.env_base import BaseEnv
from envs.JSBSim.envs.singlecontrol_env_cont import SingleControlEnv
from envs.JSBSim.envs.singlecontrol_env_cont_missile import SingleControlMissileEnv
from envs.JSBSim.envs.singlecombat_env import SingleCombatEnv
from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv
from envs.JSBSim.torchrl.tensor_specs import ConvertibleMultiOneHotDiscreteTensorSpec

__all__ = ["JSBSimWrapper", "JSBSimEnv"]

def is_single_agent_env(env: BaseEnv) -> bool: 
    return isinstance(env, SingleControlEnv) or isinstance(env, SingleCombatEnv) \
    or isinstance(env, SingleControlMissileEnv)

class JSBSimWrapper(_EnvWrapper):

    git_url: str = "https://github.com/dennismalmgren/CloseAirCombat"
    libname: str = "JSBSim"
    
    available_envs: Dict[str, Any] = {} #TODO

    """
        JSBSim Wrapper
    """
    def __init__(
            self,
            env: BaseEnv,
            categorical_action_encoding: bool = True,
            group_map: MarlGroupMapType | Dict[str, List[str]] | None = None,
            **kwargs,
    ):
        self.categorical_action_encoding = categorical_action_encoding
        if env is not None:
            kwargs["env"] = env
        if is_single_agent_env(env) and group_map is not None:
            raise ValueError(
                "group_map should be None for single agent environments."
            )
        elif not is_single_agent_env(env):
            self.group_map = group_map
        super().__init__(**kwargs, allow_done_after_reset=True)
        
    def _build_env(
            self,
            env: BaseEnv,
        ):
        #set seeds?
        # if len(self.batch_size) == 0:
        #     self.batch_size = torch.Size((1,))
        return env
    
    def _check_kwargs(self, kwargs: Dict):
        #jsbsim = self.lib

        if "env" not in kwargs:
            raise TypeError("Could not find environment key 'env' in kwargs.")
        
        env = kwargs["env"]
        if not isinstance(env, BaseEnv):
            raise TypeError(
                "env is not of type 'JSBSim.envs.env_base.BaseEnv'."
            )
        
    def _init_env(self):
        pass
        #self.reset()

    def _get_default_group_map(self, agent_names: List[str]):
        #This assumes that the first letter indicates the team
        team_id_s = set([agent_name[0] for agent_name in agent_names])
        group_map = {
            team_id: [agent_name for agent_name in agent_names if agent_name[0] == team_id]
            for team_id in team_id_s
        }
        return group_map
    
    def _make_specs(
        self, env: BaseEnv
    ) -> None:
        if is_single_agent_env(env):
            self.action_spec = CompositeSpec(
                    {
                        "action": self._jsbsim_to_torchrl_spec_transform(
                                        self.action_space, 
                                        self.device, 
                                        self.categorical_action_encoding
                                )
                    }
                )
            self.observation_spec = CompositeSpec(
                    {
                        "observation": self._jsbsim_to_torchrl_spec_transform(
                                        self.observation_space, 
                                        self.device, 
                                        self.categorical_action_encoding
                                )
                    }
                )
            self.reward_spec = CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            shape=torch.Size((1,)), 
                            device=self.device,
                            dtype=torch.float32
                        )
                    }
                )
            
            self.done_spec = CompositeSpec(
                {
                    "done": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((1,)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "terminated": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((1,)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                    "truncated": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((1,)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                },
            )
        else:
            self.agent_names = [key for key in env._jsbsims.keys()]
            self.agent_names_to_indices_map = {
                uid: i for i, uid in enumerate(env.ego_ids + env.enm_ids)
            }

            if self.group_map is None:
                self.group_map = self._get_default_group_map(self.agent_names)
            elif isinstance(self.group_map, MarlGroupMapType):
                self.group_map = self.group_map.get_group_map(self.agent_names)
            check_marl_grouping(self.group_map, self.agent_names)

            self.unbatched_action_spec = CompositeSpec(device=self.device)
            self.unbatched_observation_spec = CompositeSpec(device=self.device)
            self.unbatched_reward_spec = CompositeSpec(device=self.device)

            self.het_specs = False
            self.het_specs_map = {}
            for group in self.group_map.keys():
                (
                    group_observation_spec,
                    group_action_spec,
                    group_reward_spec,
                ) = self._make_unbatched_group_specs(group)
        
                self.unbatched_action_spec[group] = group_action_spec
                self.unbatched_observation_spec[group] = group_observation_spec
                self.unbatched_reward_spec[group] = group_reward_spec
                group_het_specs = isinstance(
                    group_observation_spec, LazyStackedCompositeSpec
                ) or isinstance(group_action_spec, LazyStackedCompositeSpec)
                self.het_specs_map[group] = group_het_specs
                self.het_specs = self.het_specs or group_het_specs
            self.unbatched_done_spec = CompositeSpec(
                {
                    "done": DiscreteTensorSpec(
                        n=2,
                        shape=torch.Size((1,)),
                        dtype=torch.bool,
                        device=self.device,
                    ),
                },
            )

            self.action_spec = self.unbatched_action_spec

            self.observation_spec = self.unbatched_observation_spec

            self.reward_spec = self.unbatched_reward_spec

            self.done_spec = self.unbatched_done_spec


    def _jsbsim_to_torchrl_spec_transform(
            self,
            spec: gym.space,
            device: DEVICE_TYPING,
            categorical_action_encoding: bool
        ):
        
        if isinstance(spec, gym.spaces.Discrete):
            action_space_cls = (
                DiscreteTensorSpec
                if categorical_action_encoding
                else OneHotDiscreteTensorSpec
            )

            dtype = (
                numpy_to_torch_dtype_dict[spec.dtype]
                if categorical_action_encoding
                else torch.long
            )

            return action_space_cls(spec.n, device=device, dtype=dtype)
        elif isinstance(spec, gym.spaces.MultiDiscrete):
            action_space_cls = (
                MultiDiscreteTensorSpec
                if categorical_action_encoding
                else ConvertibleMultiOneHotDiscreteTensorSpec
            )
            
            dtype = (
                numpy_to_torch_dtype_dict[spec.dtype]
                if categorical_action_encoding
                else torch.long
            )

            return action_space_cls(spec.nvec, device=device, dtype=dtype)
        elif isinstance(spec, gym.spaces.Box):
            shape = spec.shape
            if not len(shape):
                shape = torch.Size([1])
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
            low = torch.tensor(spec.low, dtype=dtype, device=device)
            high = torch.tensor(spec.high, dtype=dtype, device=device)
            is_unbounded = low.isinf().all and high.isinf().all()
            return (
                UnboundedContinuousTensorSpec(shape, device=device, dtype=dtype)
                if is_unbounded
                else BoundedTensorSpec(low, high, shape, device=device, dtype=dtype)
            )
        else:
            raise NotImplementedError(
                f"spec of type {type(spec).__name__} is not supported."
            )
            

    def _make_unbatched_group_specs(self, group: str):
        action_specs = []
        observation_specs = []
        reward_specs = []
        for agent_name in self.group_map[group]:
            action_specs.append(
                CompositeSpec(
                    {
                        "action": self._jsbsim_to_torchrl_spec_transform(
                                        self.action_space, 
                                        self.device, 
                                        self.categorical_action_encoding
                                )
                    }
                ))
            
            observation_specs.append(
                CompositeSpec(
                    {
                        "observation": self._jsbsim_to_torchrl_spec_transform(
                                        self.observation_space, 
                                        self.device, 
                                        self.categorical_action_encoding
                                )
                    }
                ))
            
            reward_specs.append(
                CompositeSpec(
                    {
                        "reward": UnboundedContinuousTensorSpec(
                            shape=torch.Size((1,)), 
                            device=self.device,
                            dtype=torch.float32
                        )
                    }
                )
            )

            
        group_action_spec = torch.stack(
            action_specs, dim=0
        )
        group_observation_spec = torch.stack(
            observation_specs, dim=0
        )
        group_reward_spec = torch.stack(
            reward_specs, dim=0
        )

        return (
            group_observation_spec,
            group_action_spec,
            group_reward_spec,
        )
    
    def _set_seed(self, seed: Optional[int]):
        self._env.seed(seed)

    def _reset_output_transform(self,
                                reset_outputs_tuple: Tuple | np.ndarray) -> Tuple[Any, dict]:
        if is_single_agent_env(self._env):
            obs, info = reset_outputs_tuple
            obs = obs.squeeze()
            return obs, info        
        else:
            raise NotImplementedError("Not implemented for multi-agent environments.")
    
    def read_reward(self, reward):
        return self.reward_spec.encode(reward, ignore_device=True)
    
    def _reset(
        self, tensordict: Optional[TensorDictBase] = None, **kwargs
    ) -> TensorDictBase:
        if is_single_agent_env(self._env):
            obs, _ = self._reset_output_transform(self._env.reset(**kwargs))
            source = self.read_obs(obs)
            tensordict_out = TensorDict(
                source=source,
                batch_size=self.batch_size,
            )
            tensordict_out = tensordict_out.to(self.device, non_blocking=True)
            return tensordict_out
        else:
            for group, agent_names in self.group_map.items():
                agent_tds = []
                for agent_name in agent_names:
                    i = self.agent_names_to_indices_map[agent_name]
                    agent_obs = obs[i]
                    #agent_info = {} #no info right now.
                    agent_td = TensorDict(
                        source = {
                            "observation": agent_obs
                        },
                        batch_size = [], #really?
                        device=self.device
                    )
                    agent_tds.append(agent_td)
                agent_tds = torch.stack(agent_tds, dim=0)
                if not self.het_specs_map[group]:
                    agent_tds = agent_tds.to_tensordict()
                source.update({group: agent_tds})
        tensordict_out = TensorDict(
            source=source,
            batch_size = [],
            device=self.device
        )
        return tensordict_out
    
    def read_action(self, action, group=None):
        if is_single_agent_env(self._env):
            action = action.unsqueeze(0) #add agent dim
            return self.action_spec.to_numpy(action, safe=False)
        else:
            raise NotImplementedError("Not implemented for multi-agent environments.")
    
    def _output_transform(
            self, 
            step_outputs_tuple: Tuple
    ) -> Tuple[Any, 
               float | np.ndarray | None,
               bool | np.ndarray | None,
               bool | np.ndarray | None,
               bool | np.ndarray | None,
               dict]:
        """
        Must return a tuple: (obs, reward, terminated, truncated, done, info)."""
        if is_single_agent_env(self._env):
            obs, rew, terminated, truncated, info = step_outputs_tuple
            obs = obs.squeeze(0)
            rew = rew.squeeze(0)
            terminated = terminated.squeeze(0)
            truncated = truncated.squeeze(0)
            done = terminated or truncated
        else:
            raise NotImplementedError("Not implemented for multi-agent environments.")
        return obs, rew, terminated, truncated, done, info
    
    
    def read_obs(self, observations: np.ndarray) -> Dict[str, Any]:
        if is_single_agent_env(self._env):
            (key,) = itertools.islice(self.observation_spec.keys(True, True), 1)
            observations = {key: observations}
            for key, val in observations.items():
                observations[key] = self.observation_spec[key].encode(val, ignore_device=True)

        else:
            raise NotImplementedError("Not implemented for multi-agent environments.")
        return observations

    def read_done(self, terminated: np.ndarray, truncated: np.ndarray, done: np.ndarray) -> Tuple[torch.Tensor]:
        return self.done_spec["terminated"].encode(terminated, ignore_device=True), \
                self.done_spec["truncated"].encode(truncated, ignore_device=True), \
                self.done_spec["done"].encode(done, ignore_device=True)
                
                
    def read_reward(self, reward: np.ndarray) -> np.ndarray:
        return self.reward_spec.encode(reward, ignore_device=True)
    
    def _step(
            self,
            tensordict: TensorDictBase,
    ) -> TensorDictBase:
        if is_single_agent_env(self._env):
            #tadaaa!
            action = tensordict.get(self.action_key)
            action_np = self.read_action(action)
            obs, reward, terminated, truncated, done, info = self._output_transform(self._env.step(action_np))
            source = self.read_obs(obs) #is now a dict.
            terminated, truncated, done = self.read_done(terminated, truncated, done)
            reward = self.read_reward(reward)

            source.update({self.reward_key: reward})
            source.update({"done": done})
            source.update({"terminated": terminated})
            source.update({"truncated": truncated})
            source.update({"cruise_missile_event_reward": torch.tensor([info["cruise_missile_event_reward"]])})
            tensordict_out = TensorDict(
                source = source,
                batch_size = tensordict.batch_size, 
                device=self.device
            )
            #TODO: support infodictreader
            return tensordict_out
          
        else:
            agent_indices = {}
            action_list = []
            n_agents = 0

            for group, agent_names in self.group_map.items():
                group_action = tensordict.get((group, "action"))
                agent_actions = list(self.read_action(group_action, group=group))
                agent_indices.update(
                    {
                        self.agent_names_to_indices_map[agent_name]: i + n_agents
                        for i, agent_name in enumerate(agent_names)
                    }
                )
                n_agents += len(agent_names)
                action_list += agent_actions
            action = [action_list[agent_indices[i]] for i in range(self.n_agents)]

            obs, rews, dones, infos = self._env.step(action)    
            source = {"done": dones[0], "terminated": dones[0].clone()}
            for group, agent_names in self.group_map.items():
                agent_tds = []
                for agent_name in agent_names:
                    i = self.agent_names_to_indices_map[agent_name]
                    agent_obs = obs[i] #recursive?
                    agent_rew = rews[i]
    #                agent_done = dones[i]
    #                agent_info = infos[i]
                    agent_td = TensorDict(
                        source = {
                            "observation": agent_obs,
                            "reward": agent_rew,                      
                        },
                        batch_size = [], #really?
                        device=self.device
                    )
                    agent_tds.append(agent_td)
                agent_tds = torch.stack(agent_tds, dim=1)
                if not self.het_specs_map[group]:
                    agent_tds = agent_tds.to_tensordict()
                source.update({group: agent_tds})
        tensordict_out = TensorDict(
            source = source,
            batch_size = [],
            device=self.device,
        )
        return tensordict_out

