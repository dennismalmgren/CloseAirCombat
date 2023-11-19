import sys
import os
import torch
import pytest
import numpy as np
from itertools import product
import gymnasium.spaces
from torchrl.envs.utils import check_env_specs
from torchrl.collectors import RandomPolicy
from torchrl.envs import (
    EnvCreator,
    ParallelEnv,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from envs.JSBSim.torchrl.air_combat_env_wrapper import JSBSimWrapper

from envs.JSBSim.envs.singlecontrol_env import SingleControlEnv


class TestJSBSim:
#can we support continuous actions?
#todo: move time limit to wrapper
    #@pytest.mark.parametrize("scenario_name", JSBSimWrapper.available_envs)
    #@pytest.mark.parametrize("continuous_actions", [True, False])
    def test_jsbsim_single_scenario(self):
        env = SingleControlEnv("1/heading")
        wrapped_env = JSBSimWrapper(env)
        assert wrapped_env.num_agents == 1
        for agent in env.agents.values():
            assert len(agent.partners) == 0
            assert len(agent.enemies) == 0

        check_env_specs(wrapped_env, seed=42)

    def test_jsbsim_single_scenario_collect(self):
        env = SingleControlEnv("1/heading")
        wrapped_env = JSBSimWrapper(env)
        actor = RandomPolicy(wrapped_env.action_spec)
        td = wrapped_env.rollout(100, actor)

    def test_parallelenv(self):
        num_workers = 5
        vec_env = ParallelEnv(
            create_env_fn=EnvCreator(lambda: JSBSimWrapper(SingleControlEnv("1/heading"))),
            num_workers=num_workers,
        )    

        actor = RandomPolicy(vec_env.action_spec) #or is it the inner space we need? probably..
    
        td = vec_env.rollout(100, actor)
        print('ok')