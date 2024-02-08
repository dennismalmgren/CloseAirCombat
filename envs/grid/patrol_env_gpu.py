import os
import sys
from typing import Tuple, TypeVar, Any, Optional
from enum import Enum
import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.patrol_reachability import calc_distance_matrix

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Dir(Enum):
    N = 0
    E = 1
    S = 2
    W = 3

class PatrolEnvGpu(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"]
    }

    def __init__(self, 
                 render_mode: Optional[str] = None):
        self.device = "cuda"
        self.width = torch.tensor(100, device=self.device)
        self.height = torch.tensor(50, device=self.device)
        self.size = self.width * self.height
        self.dir = torch.tensor(1, device=self.device)
        self.loc_h = torch.tensor(0, device=self.device)
        self.loc_w = torch.tensor(0, device=self.device)
        self.visitation = torch.tensor(1, device=self.device)
        self.dir_scale = torch.tensor(1 / 4, device=self.device)
        self.top_row_index = torch.tensor(0, device=self.device)
        self.bottom_row_index = self.height - 1
        self.left_col_index = torch.tensor(0, device=self.device)
        self.right_col_index = self.width - 1
        self.action_mask = torch.ones(3, dtype=torch.bool, device=self.device)
        self.available_action = torch.tensor(1, dtype=torch.bool, device=self.device)
        self.unavailable_action = torch.tensor(0, dtype=torch.bool, device=self.device)

        self.observation_mode = "plain"
        if self.observation_mode == "pixels":
            #First we give the grid. This has intensity values.
            #Then we give our location. That has a direction value (0.25, 0.5, 0.75, 1.0)
            #Finally we give the reachability from our location. This is a distance value.
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.height, self.width, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 + self.height.item() * self.width.item(),))
        self.action_space = spaces.Discrete(3) #straight, turn left, turn right
        self.dir_move_map_tensor = torch.tensor([
            [-1, 0],  # (Dir.N, 0)
            [0, -1],  # (Dir.N, 1)
            [0, 1],   # (Dir.N, 2)
            [0, 1],   # (Dir.E, 0)
            [-1, 0],  # (Dir.E, 1)
            [1, 0],   # (Dir.E, 2)
            [1, 0],   # (Dir.S, 0)
            [0, 1],   # (Dir.S, 1)
            [0, -1],  # (Dir.S, 2)
            [0, -1],  # (Dir.W, 0)
            [1, 0],   # (Dir.W, 1)
            [-1, 0]   # (Dir.W, 2)
        ], dtype=torch.int, device=self.device)

        self.dir_turn_map_tensor = torch.tensor([
            0,  # (Dir.N, 0)
            3,  # (Dir.N, 1)
            1,  # (Dir.N, 2)
            1,  # (Dir.E, 0)
            0,  # (Dir.E, 1)
            2,  # (Dir.E, 2)
            2,  # (Dir.S, 0)
            1,  # (Dir.S, 1)
            3,  # (Dir.S, 2)
            3,  # (Dir.W, 0)
            2,  # (Dir.W, 1)
            0   # (Dir.W, 2)
        ], dtype=torch.int, device=self.device)
        self.dir_mask_map_h_top_tensor = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
        ], dtype=torch.bool, device=self.device)
        self.dir_mask_map_h_bottom_tensor = torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ], dtype=torch.bool, device=self.device)

        self.dir_mask_map_w_left_tensor = torch.tensor([
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=torch.bool, device=self.device)

        self.dir_mask_map_w_right_tensor = torch.tensor([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=torch.bool, device=self.device)
        
        self._calculate_dists()
        self._reset()    
        self.render_mode = render_mode


    def _calculate_dists(self):
        #dists are a matrix loc_h, loc_w, dir to loc_h, loc_w.
        #actions/edges are forward, left, right 
        #we are working with opportunity costs, so closest path indicates reward.
        dist_matrix = calc_distance_matrix(self.height.item(), self.width.item())
        self.dist_matrix = torch.tensor(dist_matrix, device=self.device)

        self.dist_matrix_no_dir = self.dist_matrix.min(-1)[0]
#        np.save("dist_matrix.npy", self.dist_matrix)
#        np.save("dist_matrix_no_dir.npy", self.dist_matrix_no_dir)

    def _update_state_history(self):
        self.state_history[self.loc_h, self.loc_w] += self.visitation

    def _reset(self):
        self.loc_h = torch.tensor(0, device=self.device)
        self.loc_w = torch.tensor(0, device=self.device)
        self.dir = torch.tensor(1, device=self.device)
        self.expected_arrivals_grid = torch.ones((self.height, self.width), dtype=torch.float32, device=self.device) * 1 / 100.0 #uniform
        self.state_history = torch.zeros((self.height, self.width), dtype=torch.float32, device=self.device)
        self._update_state_history()

    def _calculate_reward(self):
        reward_tot = torch.sum(self.dist_matrix_no_dir[self.loc_h, self.loc_w, self.dir] * self.expected_arrivals_grid)
        #return the mean expected reachability
        #we can do cvar here.
        return -reward_tot / self.size
    

    def _loc1d(self) -> int:
        return self.loc_h * self.width + self.loc_w
    
    def reset(self, 
              *, 
              seed: Optional[int] = None,
              options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed = seed)
        self._reset()
        obs = self._create_obs()
        action_mask = self._action_mask()
        info = {
            "action_mask": action_mask
        }
        if self.render_mode == "human":
            self.render()
        return obs, info

    def _create_obs(self) -> ObsType:
        if self.observation_mode == "pixels":
            returned_grid = np.copy(self.expected_arrivals_grid)
            returned_loc = np.zeros((self.height, self.width), dtype = np.float32)
            returned_loc[self.loc_h, self.loc_w] =  1 / (self.dir.value + 1)
            returned_reach = self.dist_matrix_no_dir[self.loc_h, self.loc_w, self.dir.value] / np.max(np.nan_to_num(self.dist_matrix_no_dir))
            obs = np.stack((returned_grid, returned_loc, returned_reach), axis = -1)
            return obs
        else:
            returned_grid = self.expected_arrivals_grid
            dir_obs = (self.dir * self.dir_scale).reshape(1)
            loc_h_obs = (self.loc_h / self.height).reshape(1)
            loc_w_obs = (self.loc_w / self.width).reshape(1)

            obs = torch.cat((dir_obs, loc_h_obs, loc_w_obs, returned_grid.flatten()))

            return obs
    

    def _action_mask(self):
        self.action_mask[:] = self.available_action
        if self.loc_h == self.top_row_index:
            self.action_mask &= ~self.dir_mask_map_h_top_tensor[self.dir]
        elif self.loc_h == self.bottom_row_index:
            self.action_mask &= ~self.dir_mask_map_h_bottom_tensor[self.dir]

        if self.loc_w == self.left_col_index:
            self.action_mask &= ~self.dir_mask_map_w_left_tensor[self.dir]
        elif self.loc_w == self.right_col_index:
            self.action_mask &= ~self.dir_mask_map_w_right_tensor[self.dir]
        return self.action_mask
        
    def _move_from(self, loc_h: int, loc_w: int, dir, action) -> Tuple[float, float, Dir]:
        index = self.dir * 3 + action
        moves = self.dir_move_map_tensor[index]
        next_dir = self.dir_turn_map_tensor[index]
        return loc_h + moves[0], loc_w + moves[1], next_dir

    def _move(self, action: ActType) -> Tuple[float, float, Dir]:
        #We assume it's legal.
        return self._move_from(self.loc_h, self.loc_w, self.dir, action)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict[str, Any]]:
        action = torch.tensor(action, device=self.device)
        action_mask = self._action_mask()
        if action_mask[action] == 0:
            raise Exception(f"Invalid action sent: {action}")
    
        next_loc_h, next_loc_w, next_dir = self._move(action)    
        self.loc_h = next_loc_h
        self.loc_w = next_loc_w    
        self.dir = next_dir
        self._update_state_history()
        reward = self._calculate_reward()
        obs = self._create_obs()
        action_mask = self._action_mask()
        info = {
            "action_mask": action_mask
        }
        if self.render_mode == "human":
            self.render()
        return (obs, reward, False, False, info)


    def render(self, mode='rgb_array'):
        return (np.copy(self.expected_arrivals_grid), np.copy(self.state_history), (self.loc_h, self.loc_w))
    
    def close(self):
        # Clean up any resources used by the environment
        pass
