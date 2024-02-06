import torch

from torchrl.modules import (
    ActorValueOperator,
    ConvNet,
    MLP,
    OneHotCategorical,
    MaskedOneHotCategorical,
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
from tensordict.nn import TensorDictModule

from torchrl.envs import (
    GymWrapper, TransformedEnv, RewardSum, 
    StepCounter, Compose, default_info_dict_reader, 
    RewardScaling, step_mdp, ActionMask
)
from torchrl.data import BinaryDiscreteTensorSpec

from envs.grid.patrol_env import PatrolEnv

def make_base_env(render_mode: str = None):
    env = PatrolEnv(render_mode)
    env = GymWrapper(env, categorical_action_encoding=True)
    env.set_info_dict_reader(default_info_dict_reader(
        keys=["action_mask"],
        spec=[BinaryDiscreteTensorSpec(n=env.action_spec.space.n, dtype=torch.bool)]
    ))

    env = TransformedEnv(env,
                         Compose(
                            ActionMask(),
                            RenameTransform(in_keys=["action_mask"], out_keys=["mask"], create_copy=True),
                            StepCounter(max_steps=1000),
                            RewardSum(),
                            RewardScaling(loc=0, scale=0.01),
                         ))
    return env


def make_parallel_env(num_envs, device, is_test=False):
    env = ParallelEnv(
        num_envs,
        EnvCreator(lambda: make_base_env()),
        serial_for_single=True,
        device=device,
    )
    env = TransformedEnv(env)
  
    #We need the mask both on the parallel and on the individual environment. The individual environment is
    #called/created by envcreator, and so needs a mask, but the mask info is not propagated to the parallelenv.
    env.append_transform(ActionMask())
#    env.append_transform(StepCounter(max_steps=1000))

#    env.append_transform(RenameTransform(in_keys=["action_mask"], out_keys=["mask"], create_copy=True))
    return env

def make_ppo_modules(proof_environment):
    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, DiscreteBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = MaskedCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.low,
            "max": proof_environment.action_spec.space.high,
        }

    # Define input keys
    in_keys = ["observation"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    # common_cnn = ConvNet(
    #     activation_class=torch.nn.ReLU,
    #     num_cells=[32, 64, 64],
    #     kernel_sizes=[8, 4, 3],
    #     strides=[4, 2, 1],
    # )
    common_input = torch.ones(input_shape)
    # common_cnn_output = common_cnn(torch.ones(input_shape))
    common_mlp = MLP(
        in_features=input_shape[-1],
        #in_features=common_cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=512,
        num_cells=[],
    )
    common_mlp_output = common_mlp(common_input)

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=common_mlp,
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=common_mlp_output.shape[-1],
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
        in_features=common_mlp_output.shape[-1],
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module

def make_ppo_models():
    proof_environment = make_parallel_env(1, device="cpu")

    common_module, policy_module, value_module = make_ppo_modules(
        proof_environment
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()

    del proof_environment

    return actor, critic

def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=False,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards.append(reward)
    del td_test
    return torch.cat(test_rewards, 0).mean()