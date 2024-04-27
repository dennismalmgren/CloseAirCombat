import os
import sys
from typing import Tuple, TypeVar, Any, Optional
from enum import Enum
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from envs.grid.patrol_reachability import calc_distance_matrix

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class Dir(Enum):
    N = 0
    E = 1
    S = 2
    W = 3

class PatrolEnv(gym.Env):
    metadata = {
        "render_modes": ["rgb_array"]
    }

    def __init__(self, 
                 render_mode: Optional[str] = None):
        self.width: int = 100
        self.height: int = 50
        self.size: int = self.width * self.height
        self.dir: Dir = Dir.E
        self.loc_h: int = 0
        self.loc_w: int = 0
        self.prev_loc_h: int = 0
        self.prev_loc_w: int = 0

        self.loc_obs = np.zeros((self.height, self.width), dtype = np.float32)
        self._update_loc_obs()

        self.observation_mode = "pixels"
        if self.observation_mode == "pixels":
            #First we give the grid. This has intensity values.
            #Then we give our location. That has a direction value (0.25, 0.5, 0.75, 1.0)
            #Finally we give the reachability from our location. This is a distance value.
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(3, self.height, self.width), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3 + self.height * self.width,))
        self.action_space = spaces.Discrete(3) #straight, turn left, turn right
        self.dir_move_map = {
            (Dir.N, 0): (-1, 0),
            (Dir.N, 1): (0, -1),
            (Dir.N, 2): (0, 1),
            (Dir.E, 0): (0, 1),
            (Dir.E, 1): (-1, 0),
            (Dir.E, 2): (1, 0),
            (Dir.S, 0): (1, 0),
            (Dir.S, 1): (0, 1),
            (Dir.S, 2): (0, -1),
            (Dir.W, 0): (0, -1),
            (Dir.W, 1): (1, 0),
            (Dir.W, 2): (-1, 0)
        }
        self.dir_turn_map = {
            (Dir.N, 0): Dir.N,
            (Dir.N, 1): Dir.W,
            (Dir.N, 2): Dir.E,
            (Dir.E, 0): Dir.E,
            (Dir.E, 1): Dir.N,
            (Dir.E, 2): Dir.S,
            (Dir.S, 0): Dir.S,
            (Dir.S, 1): Dir.E,
            (Dir.S, 2): Dir.W,
            (Dir.W, 0): Dir.W,
            (Dir.W, 1): Dir.S,
            (Dir.W, 2): Dir.N
        }
        
        self._calculate_dists()
        self._reset()    
        self.render_mode = render_mode

    def _update_loc_obs(self):
        self.loc_obs[self.prev_loc_h, self.prev_loc_w] = 0
        self.loc_obs[self.loc_h, self.loc_w] = 1 / (self.dir.value + 1)

    def _calculate_dists(self):
        #dists are a matrix loc_h, loc_w, dir to loc_h, loc_w.
        #actions/edges are forward, left, right 
        #we are working with opportunity costs, so closest path indicates reward.
        self.dist_matrix = np.ones((self.height, self.width, 4, self.height, self.width, 4)) * np.inf
        self.dist_matrix = calc_distance_matrix(self.height, self.width)

        self.dist_matrix_no_dir = self.dist_matrix.min(-1)
        self.dist_matrix_no_dir = self.dist_matrix_no_dir / np.max(np.nan_to_num(self.dist_matrix_no_dir))

#        np.save("dist_matrix.npy", self.dist_matrix)
#        np.save("dist_matrix_no_dir.npy", self.dist_matrix_no_dir)

    def _update_state_history(self):
        self.state_history[self.loc_h, self.loc_w] += 1

    def _reset(self):
        self.loc_h = 0
        self.loc_w = 0
        self.prev_loc_h = 0
        self.prev_loc_w = 0
        self.dir = Dir.E
        self.expected_arrivals_grid = np.ones((self.height, self.width), dtype=np.float32) * 1 / 100.0 #uniform
        self.state_history = np.zeros((self.height, self.width), dtype=np.int32)
        self.loc_obs = np.zeros((self.height, self.width), dtype = np.float32)
        self._update_loc_obs()
        self.loc_obs[self.loc_h, self.loc_w] = 1 / (self.dir.value + 1)
        self._update_state_history()

    def _calculate_reward(self):
        reward_tot = np.sum(self.dist_matrix_no_dir[self.loc_h, self.loc_w, self.dir.value] * self.expected_arrivals_grid)
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
            returned_grid = self.expected_arrivals_grid
            returned_reach = self.dist_matrix_no_dir[self.loc_h, self.loc_w, self.dir.value]
            obs = np.stack((self.expected_arrivals_grid, self.loc_obs, returned_reach), axis = 0)
            return obs
        else:
            returned_grid = np.copy(self.expected_arrivals_grid)
            self_dir = self.dir.value / 4
            
            obs = np.concatenate((np.asarray([self.loc_h / self.height, self.loc_w / self.width, self_dir]),
                                returned_grid.flatten().astype(np.float32)))
            return obs
    
    def _action_mask_for(self, loc_h: int, loc_w: int, dir: Dir) -> np.ndarray:
        action_mask = np.ones(3, dtype=np.bool_)
        # Define a dictionary to map direction and position to action mask indices
        dir_mask_map_h = {
            (Dir.N, 'top'):    [0],
            (Dir.E, 'top'):    [1],
            (Dir.W, 'top'):    [2],
            (Dir.S, 'bottom'): [0],
            (Dir.E, 'bottom'): [2],
            (Dir.W, 'bottom'): [1],
        }

        dir_mask_map_w = {
            (Dir.N, 'left'):   [1],
            (Dir.S, 'left'):   [2],
            (Dir.W, 'left'):   [0],
            (Dir.N, 'right'):  [2],
            (Dir.S, 'right'):  [1],
            (Dir.E, 'right'):  [0]
        }
        # Determine position (top, bottom, left, right)
        h_position = None
        w_position = None
        if loc_h == 0:
            h_position = 'top'
        elif loc_h == self.height - 1:
            h_position = 'bottom'
        if loc_w == 0:
            w_position = 'left'
        elif loc_w == self.width - 1:
            w_position = 'right'

        # Set action mask based on direction and position
        if h_position:
            indices = dir_mask_map_h.get((dir, h_position), [])
            for index in indices:
                action_mask[index] = 0
        if w_position:
            indices = dir_mask_map_w.get((dir, w_position), [])
            for index in indices:
                action_mask[index] = 0
        return action_mask

    def _action_mask(self):
        return self._action_mask_for(self.loc_h, self.loc_w, self.dir)
    
    def _move_from(self, loc_h: int, loc_w: int, dir: Dir, action: int) -> Tuple[float, float, Dir]:
        moves = self.dir_move_map[(dir, action)]
        next_dir = self.dir_turn_map[(dir, action)]
        return loc_h + moves[0], loc_w + moves[1], next_dir

    def _move(self, action: ActType) -> Tuple[float, float, Dir]:
        #We assume it's legal.
        return self._move_from(self.loc_h, self.loc_w, self.dir, action)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict[str, Any]]:
        action_mask = self._action_mask()
        if action_mask[action] == 0:
            raise Exception(f"Invalid action sent: {action}")
    
        next_loc_h, next_loc_w, next_dir = self._move(action)
        self.prev_loc_h = self.loc_h
        self.prev_loc_w = self.loc_w    
        self.loc_h = next_loc_h
        self.loc_w = next_loc_w    
        self.dir = next_dir
        self._update_loc_obs()

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

