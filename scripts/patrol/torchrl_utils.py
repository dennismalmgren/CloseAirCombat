import torch.nn
import torch.optim
from torchrl.data import CompositeSpec
from torchrl.envs import RewardSum, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, QValueActor

def make_dqn_modules(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module
    mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=num_outputs,
        num_cells=[120, 84],
    )

    qvalue_module = QValueActor(
        module=mlp,
        action_mask_key="action_mask",
        spec=CompositeSpec(action=action_spec),
        in_keys=["observation"],
    )
    return qvalue_module

def make_dqn_model(env):
    qvalue_module = make_dqn_modules(env)
    return qvalue_module

def eval_model(actor, test_env, num_episodes=3):
    test_rewards = torch.zeros(num_episodes, dtype=torch.float32)
    for i in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards[i] = reward.sum()
    del td_test
    return test_rewards.mean()