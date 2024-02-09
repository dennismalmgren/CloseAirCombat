import os
import sys
from torchrl.envs.utils import check_env_specs

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.patrol_env_torchrl import PatrolEnv

def main():
    print("Testing single env, cpu")
    env = PatrolEnv(device="cpu")
    check_env_specs(env)

    print("Testing batched env, cpu")   
    batch_size = 2
    env = PatrolEnv(device="cpu", batch_size=[batch_size])
    check_env_specs(env)

    print("Testing single env, cuda")
    env = PatrolEnv(device="cuda")
    check_env_specs(env)

    print("Testing batched env, cuda")   
    batch_size = 2
    env = PatrolEnv(device="cuda", batch_size=[batch_size])
    check_env_specs(env)

    # print("observation_spec:", env.observation_spec)
    # print("state_spec:", env.state_spec)
    # print("reward_spec:", env.reward_spec)
    td = env.reset()
    print("reset tensordict", td)
    td = env.rand_step(td)
    print("random step tensordict", td)

if __name__ == "__main__":
    main()

