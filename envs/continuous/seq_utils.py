
from typing import Tuple, Dict, Any
import copy

from tensordict import TensorDictBase
import torch
from tensordict import pad, set_lazy_legacy
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data import Storage, TensorStorage
from tensordict.utils import NestedKey
from torchrl._utils import _replace_last

@set_lazy_legacy(False)
def split_trajectories_pad_left(
    rollout_tensordict: TensorDictBase, prefix=None,
    desired_length: int = None
) -> TensorDictBase:
    """A util function for trajectory separation.

    Takes a tensordict with a key traj_ids that indicates the id of each trajectory.

    From there, builds a B x T x ... zero-padded tensordict with B batches on max duration T

    Args:
        rollout_tensordict (TensorDictBase): a rollout with adjacent trajectories
            along the last dimension.
        prefix (str or tuple of str, optional): the prefix used to read and write meta-data,
            such as ``"traj_ids"`` (the optional integer id of each trajectory)
            and the ``"mask"`` entry indicating which data are valid and which
            aren't. Defaults to ``None`` (no prefix).
    """
    sep = ".-|-."

    if isinstance(prefix, str):
        traj_ids_key = (prefix, "traj_ids")
        mask_key = (prefix, "mask")
    elif isinstance(prefix, tuple):
        traj_ids_key = (*prefix, "traj_ids")
        mask_key = (*prefix, "mask")
    elif prefix is None:
        traj_ids_key = "traj_ids"
        mask_key = "mask"
    else:
        raise NotImplementedError(f"Unknown key type {type(prefix)}.")

    traj_ids = rollout_tensordict.get(traj_ids_key, None)
    done = rollout_tensordict.get(("next", "done"))
    if traj_ids is None:
        idx = (slice(None),) * (rollout_tensordict.ndim - 1) + (slice(None, -1),)
        done_sel = done[idx]
        pads = [1, 0]
        pads = [0, 0] * (done.ndim - rollout_tensordict.ndim) + pads
        done_sel = torch.nn.functional.pad(done_sel, pads)
        if done_sel.shape != done.shape:
            raise RuntimeError(
                f"done and done_sel have different shape {done.shape} - {done_sel.shape} "
            )
        traj_ids = done_sel.cumsum(rollout_tensordict.ndim - 1)
        traj_ids = traj_ids.squeeze(-1)
        if rollout_tensordict.ndim > 1:
            for i in range(1, rollout_tensordict.shape[0]):
                traj_ids[i] += traj_ids[i - 1].max() + 1
        rollout_tensordict.set(traj_ids_key, traj_ids)

    splits = traj_ids.view(-1)
    splits = [(splits == i).sum().item() for i in splits.unique_consecutive()]
    # if all splits are identical then we can skip this function
    if len(set(splits)) == 1 and splits[0] == traj_ids.shape[-1]:
        rollout_tensordict.set(
            mask_key,
            torch.ones(
                rollout_tensordict.shape,
                device=rollout_tensordict.device,
                dtype=torch.bool,
            ),
        )
        if rollout_tensordict.ndimension() == 1:
            rollout_tensordict = rollout_tensordict.unsqueeze(0)
        return rollout_tensordict.unflatten_keys(sep)
    out_splits = rollout_tensordict.view(-1).split(splits, 0)

    for out_split in out_splits:
        out_split.set(
            mask_key,
            torch.ones(
                out_split.shape,
                dtype=torch.bool,
                device=out_split.get(("next", "done")).device,
            ),
        )
    if len(out_splits) > 1:
        MAX = max(*[out_split.shape[0] for out_split in out_splits])
    else:
        MAX = out_splits[0].shape[0]
    if desired_length is not None:
        MAX = max(MAX, desired_length)
    td = torch.stack(
        [pad(out_split, [MAX - out_split.shape[0], 0]) for out_split in out_splits], 0
    ).contiguous()
    # td = td.unflatten_keys(sep)
    return td

class OnlineSliceSampler(Sampler):
    """Samples slices of data along the first dimension, given start and stop signals.

    This class samples sub-trajectories with replacement. For a version without
    replacement, see :class:`~torchrl.data.replay_buffers.samplers.SliceSamplerWithoutReplacement`.

    Keyword Args:
        num_slices (int): the number of slices to be sampled. The batch-size
            must be greater or equal to the ``num_slices`` argument. Exclusive
            with ``slice_len``.
        slice_len (int): the length of the slices to be sampled. The batch-size
            must be greater or equal to the ``slice_len`` argument and divisible
            by it. Exclusive with ``num_slices``.
        end_key (NestedKey, optional): the key indicating the end of a
            trajectory (or episode). Defaults to ``("next", "done")``.
        traj_key (NestedKey, optional): the key indicating the trajectories.
            Defaults to ``"episode"`` (commonly used across datasets in TorchRL).
        ends (torch.Tensor, optional): a 1d boolean tensor containing the end of run signals.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        trajectories (torch.Tensor, optional): a 1d integer tensor containing the run ids.
            To be used whenever the ``end_key`` or ``traj_key`` is expensive to get,
            or when this signal is readily available. Must be used with ``cache_values=True``
            and cannot be used in conjunction with ``end_key`` or ``traj_key``.
        cache_values (bool, optional): to be used with static datasets.
            Will cache the start and end signal of the trajectory.
        truncated_key (NestedKey, optional): If not ``None``, this argument
            indicates where a truncated signal should be written in the output
            data. This is used to indicate to value estimators where the provided
            trajectory breaks. Defaults to ``("next", "truncated")``.
            This feature only works with :class:`~torchrl.data.replay_buffers.TensorDictReplayBuffer`
            instances (otherwise the truncated key is returned in the info dictionary
            returned by the :meth:`~torchrl.data.replay_buffers.ReplayBuffer.sample` method).
        strict_length (bool, optional): if ``False``, trajectories of length
            shorter than `slice_len` (or `batch_size // num_slices`) will be
            allowed to appear in the batch.
            Be mindful that this can result in effective `batch_size`  shorter
            than the one asked for! Trajectories can be split using
            :func:`~torchrl.collectors.split_trajectories`. Defaults to ``True``.

    .. note:: To recover the trajectory splits in the storage,
        :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` will first
        attempt to find the ``traj_key`` entry in the storage. If it cannot be
        found, the ``end_key`` will be used to reconstruct the episodes.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.replay_buffers import LazyMemmapStorage, TensorDictReplayBuffer
        >>> from torchrl.data.replay_buffers.samplers import SliceSampler
        >>> torch.manual_seed(0)
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(1_000_000),
        ...     sampler=SliceSampler(cache_values=True, num_slices=10),
        ...     batch_size=320,
        ... )
        >>> episode = torch.zeros(1000, dtype=torch.int)
        >>> episode[:300] = 1
        >>> episode[300:550] = 2
        >>> episode[550:700] = 3
        >>> episode[700:] = 4
        >>> data = TensorDict(
        ...     {
        ...         "episode": episode,
        ...         "obs": torch.randn((3, 4, 5)).expand(1000, 3, 4, 5),
        ...         "act": torch.randn((20,)).expand(1000, 20),
        ...         "other": torch.randn((20, 50)).expand(1000, 20, 50),
        ...     }, [1000]
        ... )
        >>> rb.extend(data)
        >>> sample = rb.sample()
        >>> print("sample:", sample)
        >>> print("episodes", sample.get("episode").unique())
        episodes tensor([1, 2, 3, 4], dtype=torch.int32)

    :class:`~torchrl.data.replay_buffers.SliceSampler` is default-compatible with
    most of TorchRL's datasets:

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.data.datasets import RobosetExperienceReplay
        >>> from torchrl.data import SliceSampler
        >>>
        >>> torch.manual_seed(0)
        >>> num_slices = 10
        >>> dataid = list(RobosetExperienceReplay.available_datasets)[0]
        >>> data = RobosetExperienceReplay(dataid, batch_size=320, sampler=SliceSampler(num_slices=num_slices))
        >>> for batch in data:
        ...     batch = batch.reshape(num_slices, -1)
        ...     break
        >>> print("check that each batch only has one episode:", batch["episode"].unique(dim=1))
        check that each batch only has one episode: tensor([[19],
                [14],
                [ 8],
                [10],
                [13],
                [ 4],
                [ 2],
                [ 3],
                [22],
                [ 8]])

    """

    def __init__(
        self,
        *,
        num_slices: int = None,
        slice_len: int = None,
        end_key: NestedKey | None = None,
        traj_key: NestedKey | None = None,
        ends: torch.Tensor | None = None,
        trajectories: torch.Tensor | None = None,
        cache_values: bool = False,
        truncated_key: NestedKey | None = ("next", "truncated"),
        strict_length: bool = True,
    ) -> object:
        self.num_slices = num_slices
        self.slice_len = slice_len
        self.end_key = end_key
        self.traj_key = traj_key
        self.truncated_key = truncated_key
        self.cache_values = cache_values
        self._fetch_traj = True
        self._uses_data_prefix = False
        self.strict_length = strict_length
        self._cache = {}
        if trajectories is not None:
            if traj_key is not None or end_key:
                raise RuntimeError(
                    "`trajectories` and `end_key` or `traj_key` are exclusive arguments."
                )
            if ends is not None:
                raise RuntimeError("trajectories and ends are exclusive arguments.")
            if not cache_values:
                raise RuntimeError(
                    "To be used, trajectories requires `cache_values` to be set to `True`."
                )
            vals = self._find_start_stop_traj(trajectory=trajectories)
            self._cache["stop-and-length"] = vals

        elif ends is not None:
            if traj_key is not None or end_key:
                raise RuntimeError(
                    "`ends` and `end_key` or `traj_key` are exclusive arguments."
                )
            if trajectories is not None:
                raise RuntimeError("trajectories and ends are exclusive arguments.")
            if not cache_values:
                raise RuntimeError(
                    "To be used, ends requires `cache_values` to be set to `True`."
                )
            vals = self._find_start_stop_traj(end=ends)
            self._cache["stop-and-length"] = vals

        else:
            if end_key is None:
                end_key = ("next", "done")
            if traj_key is None:
                traj_key = "episode"
            self.end_key = end_key
            self.traj_key = traj_key

        if not ((num_slices is None) ^ (slice_len is None)):
            raise TypeError(
                "Either num_slices or slice_len must be not None, and not both. "
                f"Got num_slices={num_slices} and slice_len={slice_len}."
            )

    @staticmethod
    def _find_start_stop_traj(*, trajectory=None, end=None):
        if trajectory is not None:
            # slower
            # _, stop_idx = torch.unique_consecutive(trajectory, return_counts=True)
            # stop_idx = stop_idx.cumsum(0) - 1

            # even slower
            # t = trajectory.unsqueeze(0)
            # w = torch.tensor([1, -1], dtype=torch.int).view(1, 1, 2)
            # stop_idx = torch.conv1d(t, w).nonzero()

            # faster
            end = trajectory[:-1] != trajectory[1:]
            end = torch.cat([end, torch.ones_like(end[:1])], 0)
        else:
            end = torch.index_fill(
                end,
                index=torch.tensor(-1, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if end.ndim != 1:
            raise RuntimeError(
                f"Expected the end-of-trajectory signal to be 1-dimensional. Got a {end.ndim} tensor instead."
            )
        stop_idx = end.view(-1).nonzero().view(-1)
        start_idx = torch.cat([torch.zeros_like(stop_idx[:1]), stop_idx[:-1] + 1])
        lengths = stop_idx - start_idx + 1
        return start_idx, stop_idx, lengths

    def _tensor_slices_from_startend(self, seq_length, start):
        if isinstance(seq_length, int):
            return (
                torch.arange(
                    seq_length, device=start.device, dtype=start.dtype
                ).unsqueeze(0)
                + start.unsqueeze(1)
            ).view(-1)
        else:
            # when padding is needed
            return torch.cat(
                [
                    _start
                    + torch.arange(_seq_len, device=start.device, dtype=start.dtype)
                    for _start, _seq_len in zip(start, seq_length)
                ]
            )

    def _get_stop_and_length(self, storage, fallback=True):
        if self.cache_values and "stop-and-length" in self._cache:
            return self._cache.get("stop-and-length")

        if self._fetch_traj:
            # We first try with the traj_key
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                if isinstance(storage, TensorStorage):
                    try:
                        trajectory = storage._storage.get(self._used_traj_key)
                    except KeyError:
                        trajectory = storage._storage.get(("_data", self.traj_key))
                        # cache that value for future use
                        self._used_traj_key = ("_data", self.traj_key)
                    self._uses_data_prefix = (
                        isinstance(self._used_traj_key, tuple)
                        and self._used_traj_key[0] == "_data"
                    )
                else:
                    try:
                        trajectory = storage[:].get(self.traj_key)
                    except Exception:
                        raise RuntimeError(
                            "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
                        )
                vals = self._find_start_stop_traj(trajectory=trajectory[: len(storage)])
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = False
                    return self._get_stop_and_length(storage, fallback=False)
                raise

        else:
            try:
                # In some cases, the storage hides the data behind "_data".
                # In the future, this may be deprecated, and we don't want to mess
                # with the keys provided by the user so we fall back on a proxy to
                # the traj key.
                if isinstance(storage, TensorStorage):
                    try:
                        done = storage._storage.get(self._used_end_key)
                    except KeyError:
                        done = storage._storage.get(("_data", self.end_key))
                        # cache that value for future use
                        self._used_end_key = ("_data", self.end_key)
                    self._uses_data_prefix = (
                        isinstance(self._used_end_key, tuple)
                        and self._used_end_key[0] == "_data"
                    )
                else:
                    try:
                        done = storage[:].get(self.end_key)
                    except Exception:
                        raise RuntimeError(
                            "Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories."
                        )
                vals = self._find_start_stop_traj(end=done.squeeze()[: len(storage)])
                if self.cache_values:
                    self._cache["stop-and-length"] = vals
                return vals
            except KeyError:
                if fallback:
                    self._fetch_traj = True
                    return self._get_stop_and_length(storage, fallback=False)
                raise

    def _adjusted_batch_size(self, batch_size):
        if self.num_slices is not None:
            if batch_size % self.num_slices != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the number of slices, got batch_size={batch_size} and num_slices={self.num_slices}."
                )
            seq_length = batch_size // self.num_slices
            num_slices = self.num_slices
        else:
            if batch_size % self.slice_len != 0:
                raise RuntimeError(
                    f"The batch-size must be divisible by the slice length, got batch_size={batch_size} and slice_len={self.slice_len}."
                )
            seq_length = self.slice_len
            num_slices = batch_size // self.slice_len
        return seq_length, num_slices

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        # pick up as many trajs as we need
        start_idx, stop_idx, lengths = self._get_stop_and_length(storage)
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        return self._sample_slices(lengths, start_idx, stop_idx, seq_length, num_slices)

    def _sample_slices(
        self, lengths, start_idx, stop_idx, seq_length, num_slices, traj_idx=None
    ) -> Tuple[torch.Tensor, dict]:
        if traj_idx is None:
            traj_idx = torch.randint(
                lengths.shape[0], (num_slices,), device=lengths.device
            )
        else:
            num_slices = traj_idx.shape[0]

        if (lengths < seq_length).any():
            if self.strict_length:
                raise RuntimeError(
                    "Some stored trajectories have a length shorter than the slice that was asked for. "
                    "Create the sampler with `strict_length=False` to allow shorter trajectories to appear "
                    "in you batch."
                )
            # make seq_length a tensor with values clamped by lengths
            seq_length = lengths[traj_idx].clamp_max(seq_length)

        relative_starts = (
            (
                torch.rand(num_slices, device=lengths.device)
                * (lengths[traj_idx] - seq_length + 1)
            )
            .floor()
            .to(start_idx.dtype)
        )
        starts = start_idx[traj_idx] + relative_starts
        index = self._tensor_slices_from_startend(seq_length, starts)
        if self.truncated_key is not None:
            truncated_key = self.truncated_key
            done_key = _replace_last(truncated_key, "done")
            terminated_key = _replace_last(truncated_key, "terminated")

            truncated = torch.zeros(
                (*index.shape, 1), dtype=torch.bool, device=index.device
            )
            if isinstance(seq_length, int):
                truncated.view(num_slices, -1)[:, -1] = 1
            else:
                truncated[seq_length.cumsum(0) - 1] = 1
            traj_terminated = stop_idx[traj_idx] == start_idx[traj_idx] + seq_length - 1
            terminated = torch.zeros_like(truncated)
            if traj_terminated.any():
                if isinstance(seq_length, int):
                    truncated.view(num_slices, -1)[traj_terminated] = 1
                else:
                    truncated[(seq_length.cumsum(0) - 1)[traj_terminated]] = 1
            truncated = truncated & ~terminated
            done = terminated | truncated
            return index.to(torch.long), {
                truncated_key: truncated,
                done_key: done,
                terminated_key: terminated,
            }
        return index.to(torch.long), {}

    @property
    def _used_traj_key(self):
        return self.__dict__.get("__used_traj_key", self.traj_key)

    @_used_traj_key.setter
    def _used_traj_key(self, value):
        self.__dict__["__used_traj_key"] = value

    @property
    def _used_end_key(self):
        return self.__dict__.get("__used_end_key", self.end_key)

    @_used_end_key.setter
    def _used_end_key(self, value):
        self.__dict__["__used_end_key"] = value

    def _empty(self):
        pass

    def dumps(self, path):
        # no op - cache does not need to be saved
        ...

    def loads(self, path):
        # no op
        ...

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        ...

    def __getstate__(self):
        state = copy(self.__dict__)
        state["_cache"] = {}
        return state
