import os
import sys
from typing import Tuple, TypeVar, Any, Optional, Union
from enum import Enum
import numpy as np
from torchrl.envs import EnvBase
import torch
from torchrl.data.utils import DEVICE_TYPING
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, OneHotDiscreteTensorSpec
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs import (
    CatTensors,
    EnvBase,
    Transform,
    TransformedEnv,
    UnsqueezeTransform,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.patrol_reachability import calc_distance_matrix

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Dir(Enum):
    N = 0
    E = 1
    S = 2
    W = 3
#fully gpu vectorized env.
    
class PatrolEnv(EnvBase):
    metadata = {
        "render_modes": ["rgb_array"],
    }
    batch_locked = False

    def __init__(self, 
            *,
            device: DEVICE_TYPING = None,
            batch_size: Optional[torch.Size] = None,
            seed = None
        ):
        super().__init__(device=device,
                         batch_size=batch_size)
        self.width: torch.Tensor = torch.tensor(100, device=self.device)
        self.height: torch.Tensor = torch.tensor(50, device=self.device)
        self.loc_scale = torch.tensor([1 / self.height, 1 / self.width], device=self.device)
        self.dir_scale = torch.tensor(1 / 4, device=self.device)
        self.size: torch.Tensor = self.width * self.height
        
        self.is_batched = (self.batch_size is not None and len(self.batch_size) > 0)


        #so given a direction, the action takes on a different meaning. 
        #0 is forward, 1 is left, 2 is right.
        #three actions, so move_index = action + 3 * dir
        self.move_options = torch.tensor([[-1, 0], 
                                          [0, -1], 
                                          [0,  1], 
                                          [0,  1], 
                                          [-1, 0], 
                                          [1,  0], 
                                          [1,  0], 
                                          [0,  1], 
                                          [0, -1], 
                                          [0, -1], 
                                          [1,  0], 
                                          [-1, 0]], dtype = torch.int32, device=self.device)
        
        self.turn_options = torch.tensor([[0], 
                                          [3], 
                                          [1], 
                                          [1], 
                                          [0], 
                                          [2], 
                                          [2], 
                                          [1], 
                                          [3], 
                                          [3], 
                                          [2], 
                                          [0]], dtype = torch.int32, device=self.device)

        self.agent_loc = torch.zeros((*self.batch_size, 2), dtype=torch.int32, device=self.device)
        self.agent_dir = torch.ones((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        self.action_mask = torch.ones((*self.batch_size, 3), dtype=torch.bool, device=self.device)
        self.expected_arrivals_grid = torch.ones((*self.batch_size, self.height, self.width), dtype=torch.float32) * 1 / 100.0 #uniform
        hh, ww = torch.meshgrid(torch.arange(self.height, device = self.device), torch.arange(self.width, device = self.device), indexing='ij')
        self.hh = hh
        self.ww = ww
        self._make_action_mask_filters()
        self._calculate_dists()
        self._make_specs()
        self._make_action_mask()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

    def _make_action_mask_filters(self):
        #these should be prepared beforehand.
        self.top_bottom_masks = torch.ones((3, 4, 3), dtype=torch.bool, device=self.device)

        #top
        self.top_bottom_masks[0, 0] = torch.tensor([0, 1, 1], dtype=torch.bool, device=self.device)
        self.top_bottom_masks[0, 1] = torch.tensor([1, 0, 1], dtype=torch.bool, device=self.device)
        self.top_bottom_masks[0, 2] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) #never gonna happen..
        self.top_bottom_masks[0, 3] = torch.tensor([1, 1, 0], dtype=torch.bool, device=self.device)

        #middle
        self.top_bottom_masks[1, 2] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) 
        self.top_bottom_masks[1, 2] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) 
        self.top_bottom_masks[1, 2] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device)
        self.top_bottom_masks[1, 2] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) 

        #bottom
        self.top_bottom_masks[2, 0] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) #never gonna happen..
        self.top_bottom_masks[2, 1] = torch.tensor([1, 1, 0], dtype=torch.bool, device=self.device)
        self.top_bottom_masks[2, 2] = torch.tensor([0, 1, 1], dtype=torch.bool, device=self.device)
        self.top_bottom_masks[2, 3] = torch.tensor([1, 0, 1], dtype=torch.bool, device=self.device)

        self.left_right_masks = torch.ones((3, 4, 3), dtype=torch.bool, device=self.device)
        #left
        self.left_right_masks[0, 0] = torch.tensor([1, 0, 1], dtype=torch.bool, device=self.device)
        self.left_right_masks[0, 1] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) #never gonna happen..
        self.left_right_masks[0, 2] = torch.tensor([1, 1, 0], dtype=torch.bool, device=self.device) 
        self.left_right_masks[0, 3] = torch.tensor([0, 1, 1], dtype=torch.bool, device=self.device)
        #middle
        self.left_right_masks[1, 1] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device)
        self.left_right_masks[1, 1] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device)
        self.left_right_masks[1, 1] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device)
        self.left_right_masks[1, 1] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device)
        #right
        self.left_right_masks[2, 0] = torch.tensor([1, 1, 0], dtype=torch.bool, device=self.device)
        self.left_right_masks[2, 1] = torch.tensor([0, 1, 1], dtype=torch.bool, device=self.device)
        self.left_right_masks[2, 2] = torch.tensor([1, 0, 1], dtype=torch.bool, device=self.device) 
        self.left_right_masks[2, 3] = torch.tensor([1, 1, 1], dtype=torch.bool, device=self.device) #never gonna happen..
  
    def _calculate_dists(self):
        #dists are a matrix loc_h, loc_w, dir to loc_h, loc_w.
        #actions/edges are forward, left, right 
        #we are working with opportunity costs, so closest path indicates reward.
        self.dist_matrix = torch.tensor(calc_distance_matrix(self.height.item(), self.width.item()), dtype=torch.float32, device = self.device)
        
        self.dist_matrix_no_dir = self.dist_matrix.min(-1)[0]
        self.dist_matrix_no_dir = self.dist_matrix_no_dir / torch.max(torch.nan_to_num(self.dist_matrix_no_dir))

    def _make_action_mask(self):
        top_bottom_indices = torch.ones(self.batch_size, dtype=torch.int32, device=self.device) 
        top_bottom_indices[self.agent_loc[..., -2] == 0] = 0
        top_bottom_indices[self.agent_loc[..., -2] == self.height - 1] = 2

        left_right_indices = torch.ones(self.batch_size, dtype=torch.int32, device=self.device) 
        left_right_indices[self.agent_loc[..., -1] == 0] = 0
        left_right_indices[self.agent_loc[..., -1] == self.width - 1] = 2
        
        #won't work with batching
        self.action_mask[:] = 1
        self.action_mask &= self.top_bottom_masks[top_bottom_indices, self.agent_dir.squeeze()]
        self.action_mask &= self.left_right_masks[left_right_indices, self.agent_dir.squeeze()]

    def _make_specs(self):
        self.observation_spec = CompositeSpec(
            observation = BoundedTensorSpec(
                shape=(*self.batch_size, 3 + self.height * self.width,),
                low = 0.0,
                high = 1.0,
                dtype=torch.float32
            ),
            action_mask = BoundedTensorSpec(
                shape=(*self.batch_size, 3,),
                low = 0,
                high = 1,
                dtype=torch.bool
            ),
            shape=self.batch_size
        )
    
        #maybe use one-hot.
        self.action_spec = OneHotDiscreteTensorSpec(
            n = 3,
            shape=(*self.batch_size, 3,),
            )
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1,), dtype=torch.float32)
        self.done_spec = CompositeSpec(
            terminated = BoundedTensorSpec(
                shape=(*self.batch_size, 1,),
                low = 0,
                high = 1,
                dtype=torch.bool
            ),
            done = BoundedTensorSpec(
                shape=(*self.batch_size, 1,),
                low = 0,
                high = 1,
                dtype=torch.bool
            ),
            truncated = BoundedTensorSpec(
                shape=(*self.batch_size, 1,),
                low = 0,
                high = 1,
                dtype=torch.bool
            ),
            shape=self.batch_size
        )

        #expand for batches
        # 

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        self.agent_loc = torch.zeros((*self.batch_size, 2), dtype=torch.int32, device=self.device)
        self.agent_dir = torch.ones((*self.batch_size, 1), dtype=torch.int32, device=self.device)

        self.expected_arrivals_grid = torch.ones((*self.batch_size, self.height, self.width), 
                                                    dtype=torch.float32, device = self.device) * 1 / 100.0 #uniform

        #Create observations
        scaled_loc = self.agent_loc * self.loc_scale
        scaled_dir = self.agent_dir * self.dir_scale

        observation = torch.cat((scaled_loc, scaled_dir, self.expected_arrivals_grid.reshape(*self.batch_size, -1)), dim=-1)
        terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        done = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        truncated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        self._make_action_mask()

        self.action_spec.update_mask(self.action_mask)

        out = TensorDict(
            {
                "observation": observation,
                "terminated": terminated,
                "done": done,
                "truncated": truncated,
                "action_mask": self.action_mask
            },
            batch_size=self.batch_size,
            device = self.device
        )
        return out
    

    # def _action_mask_for(self, loc_h: int, loc_w: int, dir: Dir) -> np.ndarray:
    #     action_mask = np.ones(3, dtype=np.bool_)
    #     # Define a dictionary to map direction and position to action mask indices
    #     dir_mask_map_h = {
    #         (Dir.N, 'top'):    [0],
    #         (Dir.E, 'top'):    [1],
    #         (Dir.W, 'top'):    [2],
    #         (Dir.S, 'bottom'): [0],
    #         (Dir.E, 'bottom'): [2],
    #         (Dir.W, 'bottom'): [1],
    #     }

    #     dir_mask_map_w = {
    #         (Dir.N, 'left'):   [1],
    #         (Dir.S, 'left'):   [2],
    #         (Dir.W, 'left'):   [0],
    #         (Dir.N, 'right'):  [2],
    #         (Dir.S, 'right'):  [1],
    #         (Dir.E, 'right'):  [0]
    #     }
    #     # Determine position (top, bottom, left, right)
    #     h_position = None
    #     w_position = None
    #     if loc_h == 0:
    #         h_position = 'top'
    #     elif loc_h == self.height - 1:
    #         h_position = 'bottom'
    #     if loc_w == 0:
    #         w_position = 'left'
    #     elif loc_w == self.width - 1:
    #         w_position = 'right'

    #     # Set action mask based on direction and position
    #     if h_position:
    #         indices = dir_mask_map_h.get((dir, h_position), [])
    #         for index in indices:
    #             action_mask[index] = 0
    #     if w_position:
    #         indices = dir_mask_map_w.get((dir, w_position), [])
    #         for index in indices:
    #             action_mask[index] = 0
    #     return action_mask

    # def _action_mask(self):
    #     return self._action_mask_for(self.loc_h, self.loc_w, self.dir)
    
    #use non-static version
    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        action_move_index = torch.argmax(action.float(), dim = -1) + 3 * self.agent_dir.squeeze()
        #action_move_index = action_move_index.reshape(self.batch_size)
        #this won't work with batching
        next_loc = self.agent_loc + self.move_options[action_move_index]
        next_dir = self.turn_options[action_move_index]
        self.agent_loc = next_loc
        self.agent_dir = next_dir
  
        #Create observations
        scaled_loc = self.agent_loc * self.loc_scale
        scaled_dir = self.agent_dir * self.dir_scale
        observation = torch.cat((scaled_loc, scaled_dir, self.expected_arrivals_grid.reshape(*self.batch_size, -1)), dim=-1)

        loc_dir = torch.cat((next_loc, next_dir), dim = -1)          
        if self.batch_size is not None and len(self.batch_size) > 0:  
            h_indices = loc_dir[:, 0][:, None, None]  # Shape: (B, 1, 1)
            w_indices = loc_dir[:, 1][:, None, None]  # Shape: (B, 1, 1)
            d_indices = loc_dir[:, 2][:, None, None]  # Shape: (B, 1, 1)
        else:
            h_indices = loc_dir[0]
            w_indices = loc_dir[1]
            d_indices = loc_dir[2]
        #this next does not work batched.
        reward = self.dist_matrix_no_dir[h_indices, w_indices, d_indices, self.hh, self.ww] * self.expected_arrivals_grid
        reward = torch.sum(reward, dim=-1)
        reward = -torch.sum(reward, dim=-1, keepdim=True)
        
        terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        done = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        truncated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        self._make_action_mask()
        self.action_spec.update_mask(self.action_mask)

        out = TensorDict(
            {
                "observation": observation,
                "action_mask": self.action_mask,
                "reward": reward,
                "terminated": terminated,
                "done": done,
                "truncated": truncated,
            },
            tensordict.shape,
            device = self.device
        )
        return out

    def render(self, mode='rgb_array'):
        return (np.copy(self.expected_arrivals_grid), np.copy(self.state_history), (self.loc_h, self.loc_w))
    
    def close(self):
        # Clean up any resources used by the environment
        pass

