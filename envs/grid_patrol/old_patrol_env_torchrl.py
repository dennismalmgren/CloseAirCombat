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

#from envs.grid.patrol_reachability import calc_distance_matrix

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
            render_mode: Optional[str] = None,
            device: DEVICE_TYPING = None,
            batch_size: Optional[torch.Size] = None,
            seed = None
        ):
        super().__init__(device=device,
                         batch_size=batch_size)
        self.width: torch.Tensor = torch.tensor(40, device=self.device)
        self.height: torch.Tensor = torch.tensor(20, device=self.device)
        self.loc_scale = torch.tensor([1 / self.height, 1 / self.width], device=self.device)
        self.dir_scale = torch.tensor(1 / 4, device=self.device)
        self.size: torch.Tensor = self.width * self.height
        self.render_mode = render_mode
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
        self.h_indices = torch.arange(self.height, device = self.device)
        self.w_indices = torch.arange(self.width, device = self.device)
        hh, ww = torch.meshgrid(self.h_indices, self.w_indices, indexing='ij')
        self.h_indices = self.h_indices.unsqueeze(0).unsqueeze(-1) #1 x H x 1
        self.w_indices = self.w_indices.unsqueeze(0).unsqueeze(0)  #1 x 1 x W
        self.hh = hh
        self.ww = ww

        #scenario specifics
        self.birth_rate_grid = torch.ones((*self.batch_size, self.height, self.width), dtype=torch.float32, device = self.device) * 0.1
        self.ps = torch.tensor(0.99, device = self.device)
        self.pd = torch.tensor(0.9, device = self.device)
        self.expected_arrivals_scale = (1 - self.ps) / (self.birth_rate_grid * self.size)

        self.expected_arrivals_grid = torch.zeros((*self.batch_size, self.height, self.width), dtype=torch.float32, device = self.device) 
        
        self._make_action_mask_filters()
        #self._calculate_dists()
        self._make_specs()
        self._make_action_mask()
        self._make_sensor_masks()
        if self.render_mode == "rgb_array":
            self.state_history = torch.zeros((self.height, self.width), dtype=torch.int32, device = self.device)

        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

    def _make_sensor_masks(self):
        self.sensor_width = 3 #needs to be odd
        self.sensor_range = 7
        self.sensor_halfwidth = self.sensor_width // 2
        self.sensor_area_h = torch.tensor([
            [-self.sensor_range, 0],
            [-self.sensor_halfwidth, self.sensor_halfwidth + 1],
            [1, self.sensor_range + 1],
            [-self.sensor_halfwidth, self.sensor_halfwidth + 1]
        ], dtype=torch.int32, device = self.device)

        self.sensor_area_w = torch.tensor([
            [-self.sensor_halfwidth, self.sensor_halfwidth + 1],
            [1, self.sensor_range + 1],
            [-self.sensor_halfwidth, self.sensor_halfwidth+1],
            [-self.sensor_range, 0]
        ], dtype=torch.int32, device = self.device)

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
  
    #def _calculate_dists(self):
        #dists are a matrix loc_h, loc_w, dir to loc_h, loc_w.
        #actions/edges are forward, left, right 
        #we are working with opportunity costs, so closest path indicates reward.
        #self.dist_matrix = torch.tensor(calc_distance_matrix(self.height.item(), self.width.item()), dtype=torch.float32, device = self.device)
        
        #self.dist_matrix_no_dir = self.dist_matrix.min(-1)[0]
        #self.dist_matrix_no_dir = self.dist_matrix_no_dir / torch.max(torch.nan_to_num(self.dist_matrix_no_dir))

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
                #loc (2), dir (1), ps (1), pd (1), expected arrivals (HXW), birth rates (HXW), task area (HXW), sensor coverage (HXW)
                shape=(*self.batch_size, 2 + 1 + 1 + 1 + 4 * self.height * self.width,),
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
        if self.render_mode == "":
            self.observation_spec.set("expected_arrivals_grid", 
                                      UnboundedContinuousTensorSpec(
                                            shape=(*self.batch_size, self.height, self.width,),
                                            dtype=torch.float32
                                      ))
            
            self.observation_spec.set("agent_loc",
                                      UnboundedContinuousTensorSpec(
                                            shape=(*self.batch_size, 2,),
                                            dtype=torch.float32
                                      ))
            
            self.observation_spec.set("agent_dir",
                                      UnboundedContinuousTensorSpec(
                                            shape=(*self.batch_size, 1,),
                                            dtype=torch.float32
                                      ))
            
            self.observation_spec.set("state_history",
                                        UnboundedContinuousTensorSpec(
                                            shape=(*self.batch_size, self.height, self.width,),
                                            dtype=torch.float32
                                        ))
            
            self.observation_spec.set("birth_rate_grid", 
                            UnboundedContinuousTensorSpec(
                                shape=(*self.batch_size, self.height, self.width,),
                                dtype=torch.float32
                            ))
            
            self.observation_spec.set("task_area",
                            BoundedTensorSpec(
                                shape=(*self.batch_size, self.height, self.width,),
                                low = 0,
                                high = 1,
                                dtype=torch.bool
                            ))

            self.observation_spec.set("sensor_coverage",
                            BoundedTensorSpec(
                                shape=(*self.batch_size, self.height, self.width,),
                                low = 0,
                                high = 1,
                                dtype=torch.bool
                            ))
            
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


    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _create_observation(self):
        #loc (2), dir (1), ps (1), pd (1), expected arrivals (HXW), birth rates (HXW), task area (HXW), sensor coverage (HXW)
        #todo: data types.
        scaled_loc = self.agent_loc * self.loc_scale
        scaled_dir = self.agent_dir * self.dir_scale
        observation = torch.cat((scaled_loc, scaled_dir, self.ps.expand_as(self.agent_dir), self.pd.expand_as(self.agent_dir), 
                                 self.expected_arrivals_grid.reshape(*self.batch_size, -1),
                                 self.birth_rate_grid.reshape(*self.batch_size, -1),
                                 self.task_area.reshape(*self.batch_size, -1),
                                 self.sensor_coverage.reshape(*self.batch_size, -1)
                                 ), dim=-1)
        return observation
    
    def _make_sensor_coverage(self):
        sensor_area_coverage_h = self.agent_loc[..., 0:1] + self.sensor_area_h[self.agent_dir.squeeze()] #these can probably be merged.
        sensor_area_coverage_w = self.agent_loc[..., 1:2] + self.sensor_area_w[self.agent_dir.squeeze()]

        mask_h = (self.h_indices >= sensor_area_coverage_h[..., 0:1].unsqueeze(-1)) & (self.h_indices < sensor_area_coverage_h[..., 1:2].unsqueeze(-1))
        mask_w = (self.w_indices >= sensor_area_coverage_w[..., 0:1].unsqueeze(-1)) & (self.w_indices < sensor_area_coverage_w[..., 1:2].unsqueeze(-1))
        sensor_coverage_mask = mask_h & mask_w
        self.sensor_coverage = sensor_coverage_mask & self.task_area #we can only cover our own turf.
        self.sensor_coverage = self.sensor_coverage.reshape((*self.batch_size, self.height, self.width))
        if self.render_mode == "rgb_array":
            self.sensor_coverage_render = sensor_coverage_mask.reshape((*self.batch_size, self.height, self.width))

    #separate into predict/update
    def _update_expected_arrivals(self):
        self.expected_arrivals_grid = self.expected_arrivals_grid * self.ps + self.birth_rate_grid 
        self.expected_arrivals_grid[self.sensor_coverage] *= (1 - self.pd)

    def _reset(self, tensordict: TensorDict) -> TensorDict:
        #Reset to defaults
        self.agent_loc = torch.zeros((*self.batch_size, 2), dtype=torch.int32, device=self.device)
        self.agent_dir = torch.ones((*self.batch_size, 1), dtype=torch.int32, device=self.device)
        self.task_area = torch.ones((*self.batch_size, self.height, self.width), dtype=torch.bool, device=self.device) #full mask.
        self.expected_arrivals_grid = torch.zeros((*self.batch_size, self.height, self.width), dtype=torch.float32, device = self.device) 

        self._make_sensor_coverage()

        self._update_expected_arrivals()

        observation = self._create_observation()
        terminated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        done = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        truncated = torch.zeros((*self.batch_size, 1), dtype=torch.bool, device = self.device)
        self._make_action_mask()

        self.action_spec.update_mask(self.action_mask)
        if self.render_mode == "rgb_array":
            self.state_history = torch.zeros((self.height, self.width), dtype=torch.int32, device = self.device)
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
        if self.render_mode == "rgb_array":
            self._update_state_history()
            self._render(out)
        return out
    
    def _render(self, tensordict: TensorDict):
        #for rendering, we want to add the actual sensor coverage, not only the mission relevant one.
        tensordict['expected_arrivals_grid'] = self.expected_arrivals_grid
        tensordict['agent_loc'] = self.agent_loc
        tensordict['agent_dir'] = self.agent_dir
        tensordict['state_history'] = self.state_history
        tensordict['birth_rate_grid'] = self.birth_rate_grid
        tensordict['task_area'] = self.task_area
        tensordict['sensor_coverage'] = self.sensor_coverage_render
        return tensordict
    
    def _calculate_reward(self):
        #reward is at most 1.
        return -torch.sum(torch.sum(self.expected_arrivals_grid * self.task_area * self.expected_arrivals_scale, dim = -1), dim = -1, keepdim=True)
    
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

        self._make_sensor_coverage()
        self._update_expected_arrivals()
        #Create observations
        observation = self._create_observation()

        loc_dir = torch.cat((next_loc, next_dir), dim = -1)          
        if self.batch_size is not None and len(self.batch_size) > 0:  
            h_indices = loc_dir[:, 0][:, None, None]  # Shape: (B, 1, 1)
            w_indices = loc_dir[:, 1][:, None, None]  # Shape: (B, 1, 1)
            d_indices = loc_dir[:, 2][:, None, None]  # Shape: (B, 1, 1)
        else:
            h_indices = loc_dir[0]
            w_indices = loc_dir[1]
            d_indices = loc_dir[2]
        reward = self._calculate_reward()
        
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
        if self.render_mode == "rgb_array":
            self._update_state_history()
            self._render(out)
        return out

    def _update_state_history(self):
        self.state_history[self.agent_loc[..., 0], self.agent_loc[..., 1]] += 1

    def close(self):
        # Clean up any resources used by the environment
        pass

