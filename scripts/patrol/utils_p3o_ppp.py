from typing import Optional

import torch

from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    MaskedOneHotCategorical,
    ReparamGradientStrategy,
    MaskedCategorical,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)
from torchrl.data import CompositeSpec
from torchrl.envs import (
    CatFrames,
    DoubleToFloat,
    EndOfLifeTransform,
    EnvCreator,
    ExplorationType,
    GrayScale,
    GymEnv,
    NoopResetEnv,
    ParallelEnv,
    Resize,
    RenameTransform,
    RewardClipping,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.data.tensor_specs import DiscreteBox
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.envs import (
    GymWrapper, TransformedEnv, RewardSum, 
    StepCounter, Compose, default_info_dict_reader, 
    RewardScaling, step_mdp, ActionMask
)
from torchrl.envs import ExplorationType, set_exploration_type

from torchrl.data import BinaryDiscreteTensorSpec

from envs.grid.patrol_env_ppp import PatrolEnv

def make_base_env(num_envs: int, device: str, render_mode: str):
    if num_envs > 1:
        env = PatrolEnv(device = device, render_mode = render_mode, batch_size=[num_envs])
    else:
        env = PatrolEnv(device = device, render_mode = render_mode)

    env = TransformedEnv(env,
                         Compose(
                            ActionMask(),
                            RenameTransform(in_keys=["action_mask"], out_keys=["mask"], create_copy=True),
                            StepCounter(max_steps=1000),
                            RewardScaling(loc=0, scale=0.001),
                            RewardSum(),
                         ))
    return env

#start with a baseline
def make_parallel_env(num_parallel = 1, device: str = "cpu", render_mode=None):
    return make_base_env(num_parallel, device, render_mode)
    # #Not sure you can also vectorize. We'll see.
    # env = ParallelEnv(
    #     num_parallel,
    #     EnvCreator(lambda: make_base_env("cpu", num_envs=10)),
    #     serial_for_single=True,
    #     device=device,
    # )
    # env = TransformedEnv(env)
  
    # #We need the mask both on the parallel and on the individual environment. The individual environment is
    # #called/created by envcreator, and so needs a mask, but the mask info is not propagated to the parallelenv.
    # env.append_transform(ActionMask())
    # env.append_transform(RenameTransform(in_keys=["action_mask"], out_keys=["mask"], create_copy=True)),

    # return env

class MyMaskedOneHotCategorical(MaskedOneHotCategorical):
    def __init__(
        self,
        logits: Optional[torch.Tensor] = None,
        probs: Optional[torch.Tensor] = None,
        mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        neg_inf: float = float("-inf"),
        padding_value: Optional[int] = None,
        grad_method: ReparamGradientStrategy = ReparamGradientStrategy.PassThrough,
    ) -> None:
        super().__init__(logits, probs, mask, indices, neg_inf, padding_value, grad_method)

    @property
    def mode(self):
        return torch.nn.functional.one_hot(self.probs.argmax(axis=-1), self.probs.shape[-1])
    
class CommonModule(nn.Module):
    def __init__(self, 
                 input_shape_pixels, 
                 input_shape_observation):
        super().__init__()
        self.input_shape_pixels = input_shape_pixels
        self.input_shape_observation = input_shape_observation
        
        self.pixels_cnn = ConvNet(
           # in_features= self.input_shape_pixels,
            activation_class=torch.nn.ELU,
            num_cells=[32, 32, 32],
            kernel_sizes=[3, 3, 3],
            strides=[1, 1, 1],
        )

        self.observation_mlp = MLP(
            in_features=input_shape_observation[-1],
            activation_class=torch.nn.ELU,
            activate_last_layer=True,
            out_features=512,
            num_cells=[],
        )
        
        pixels_input = torch.ones(input_shape_pixels)
        observation_input = torch.ones(input_shape_observation)
        pixels_output = self.pixels_cnn(pixels_input)
        observation_mlp_output = self.observation_mlp(observation_input)

        common_mlp_input = torch.cat(
            [
                pixels_output,
                observation_mlp_output,
            ],
            dim=-1,
        )

        self.ln = nn.LayerNorm(common_mlp_input.shape)

        self.common_mlp = MLP(
            in_features=common_mlp_input.shape[-1],
            activation_class=torch.nn.ELU,
            activate_last_layer=True,
            out_features=512,
            num_cells=[],
        )
    
    def forward(self, pixels, observation):
        pixels_output = self.pixels_cnn(pixels)
        observation_mlp_output = self.observation_mlp(observation)
        common_mlp_input = torch.cat(
                    [
                        pixels_output,
                        observation_mlp_output,
                    ],
                    dim=-1,
                )
        common_mlp_input = self.ln(common_mlp_input)
        common_mlp_output = self.common_mlp(common_mlp_input)
        return common_mlp_output
    
def make_p3o_modules(proof_environment):
    # Define input shape
    input_shape_pixels = proof_environment.observation_spec["pixels"].shape
    input_shape_observation = proof_environment.observation_spec["observation"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, DiscreteBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = MyMaskedOneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
        }

    # input from expected arrivals
    in_keys_common = ["pixels", "observation"]
    out_keys_common = ["common_features"]
    
    input_module = CommonModule(input_shape_pixels, input_shape_observation)
    pixels_input = torch.ones(input_shape_pixels)
    observation_input = torch.ones(input_shape_observation)
    common_output = input_module(pixels_input, observation_input)

    common_module = TensorDictModule(
        module=input_module,
        in_keys=in_keys_common,
        out_keys=out_keys_common,
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=common_output.shape[-1],
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits", "mask"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=common_output.shape[-1],
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module

def make_p3o_models(device):
    proof_environment = make_parallel_env(1, device=device)

    common_module, policy_module, value_module = make_p3o_modules(
        proof_environment
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    actor_critic = actor_critic.to(device)

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        #td = actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic

def eval_model(actor, test_env, num_episodes=3):
    with set_exploration_type(ExplorationType.MODE):
#        test_rewards = []
#        for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=1_000,
        )
        reward = torch.mean(td_test["next", "episode_reward"][td_test["next", "done"]])
#        test_rewards.append(reward)
    del td_test
    return reward
#    return torch.cat(test_rewards, 0).mean()