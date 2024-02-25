import brax.envs
from torchrl.envs import BraxWrapper
from torch.utils.benchmark import Timer
for batch_size in [4, 16, 128]:
    print(f"batch_size={batch_size}")
    timer = Timer('''
                  env.rollout(100)
                  ''',
        setup=f'''
        import brax.envs
        from torchrl.envs import BraxWrapper
        env = BraxWrapper(brax.envs.get_environment("ant"), batch_size=[{batch_size}])
        env.set_seed(0)
        env.rollout(2)
        ''',
    )
    print(batch_size, timer.timeit(10))
