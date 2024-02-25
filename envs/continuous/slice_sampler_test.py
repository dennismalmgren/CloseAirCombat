import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement

rb = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(1000),
    # asking for 10 slices for a total of 320 elements, ie, 10 trajectories of 32 transitions each
    sampler=SliceSamplerWithoutReplacement(num_slices=10),
    batch_size=320,
)
episode = torch.zeros(1000, dtype=torch.int)
episode[:] = 1
#episode[:300] = 1
#episode[300:550] = 2
#episode[550:700] = 3
#episode[700:] = 4
data = TensorDict(
    {
        "episode": episode,
        "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        "act": torch.randn((20,)).expand(1000, 20),
        "other": torch.randn((20, 50)).expand(1000, 20, 50),
    }, [1000]
)
rb.extend(data)
sample = rb.sample()
# since we want trajectories of 32 transitions but there are only 4 episodes to
# sample from, we only get 4 x 32 = 128 transitions in this batch
print("sample:", sample)
sample2 = rb.sample()
print("trajectories in sample", sample.get("episode").unique())