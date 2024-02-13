import sys
import os
import numpy as np
import torch
import hydra
import logging
import tqdm
import time
import copy
import tempfile

import shutil

from torchrl.envs import (
    GymWrapper, TransformedEnv, RewardSum, 
    StepCounter, Compose, default_info_dict_reader, 
    RewardScaling, step_mdp, ActionMask
)


from torchrl.collectors import RandomPolicy, SyncDataCollector
from tensordict.nn import TensorDictSequential
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from torchrl.modules import EGreedyModule
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import BinaryDiscreteTensorSpec
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patches as patches

from envs.grid.patrol_env_torchrl import PatrolEnv

from utils_ppo_gpu import make_ppo_models, make_parallel_env, eval_model, make_base_env

@hydra.main(config_path=".", config_name="config_ppo", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    start_time = time.time()
    device = "cpu" #if not torch.cuda.device_count() else "cuda"

    #rendering parameters
    L = 5 #Tail length
    N = 30 #FPS

    # Create models (check utils_atari.py)
    eval_actor, eval_critic = make_ppo_models(device)
    eval_actor, eval_critic = eval_actor.to(device), eval_critic.to(device)
    eval_actor.eval()
    run_as_debug = False
    load_from_debug = False
    #debug outputs is at the root.
    #commandline outputs is at scripts/patrol/outputs
    if run_as_debug:
        if load_from_debug:
            outputs_folder = "../../"
        else:
            outputs_folder = "../../../scripts/patrol/outputs/"
    else:
        if load_from_debug:
            outputs_folder = "../../../../../outputs"
        else:
            outputs_folder = "../../"
    
    run_id = "2024-02-13/08-25-54/"
    model_load_filename = "PPO_PPO_PatrolEnvGrid_83ae5e37_24_02_13-08_25_56_iter_final.pt"
    load_model_dir = outputs_folder + run_id + "saved_models/e3/"

    print('Loading model from ' + load_model_dir)
    loaded_state = torch.load(load_model_dir + f"{model_load_filename}")
    actor_state = loaded_state['actor']
    #critic_state = loaded_state['critic']
    #optim_state = loaded_state['optimizer']
    eval_actor.load_state_dict(actor_state)
    #critic.load_state_dict(critic_state)
    # Create a temporary directory to store the frames

    

    # Create test environment
    test_env = make_parallel_env(1, device, render_mode="rgb_array")
    test_env.eval()
    td = test_env.reset()
    episode_tds = []
    grid_history = []
    birth_history = []
    state_history = []
    loc_h_history = []
    loc_w_history = []
    sensor_coverages = []
    task_areas = []
    with set_exploration_type(ExplorationType.MODE):
        while not torch.any(td["done"]):
            td = eval_actor(td)
            td = test_env.step(td)
            #grid, state, (loc_h, loc_w) = test_env.render()
            grid_history.append(td['expected_arrivals_grid'].numpy())
            birth_history.append(td['birth_rate_grid'].numpy())
            state_history.append(td['state_history'].numpy().copy())
            loc_h_history.append(td['agent_loc'][0].item())
            loc_w_history.append(td['agent_loc'][1].item())
            sensor_coverages.append(td['sensor_coverage'].numpy().copy())
            task_areas.append(td['task_area'].numpy().copy())
            episode_tds.append(td.clone())
            td = step_mdp(td)

    episode_td = torch.stack(episode_tds).to_tensordict()
    sh_stack = np.stack(state_history)
    g_stack = np.stack(grid_history)
    b_stack = np.stack(birth_history)
    sh_max = np.max(sh_stack)
    g_max = np.max(g_stack)
    b_max = np.max(b_stack)
    T = len(episode_tds)
    temp_dir = tempfile.mkdtemp()

    H, W = grid_history[-1].shape
    # Function to create agent trajectory grid
    def create_trajectory_grid(agent_locations_h, agent_locations_w, sensor_coverages, t, L, H, W):
        grid = np.zeros((H, W))
        tail_points_h = agent_locations_h[max(0, t-L):t+1]  # Get tail points
        tail_points_w = agent_locations_w[max(0, t-L):t+1]  # Get tail points
        for point_h, point_w in zip(tail_points_h[:-1], tail_points_w[:-1]):
            grid[point_h, point_w] = 0.5  # Mark tail with intermediate intensity
        head_h = tail_points_h[-1]  # Head
        head_w = tail_points_w[-1]  # Head
        grid[head_h, head_w] = 1  # Mark head with highest intensity
        grid[sensor_coverages[t]] = 0.25 #sensor coverage
        return grid
    
    def create_task_rectangle(task_area: torch.Tensor) -> patches.Rectangle:
        border_width = 2
        y_indices, x_indices = np.where(task_area)
        # Determine the bounds of the rectangle
        y_start, y_end = np.min(y_indices), np.max(y_indices)
        x_start, x_end = np.min(x_indices), np.max(x_indices)

        # Calculate the height and width of the task area
        h_task = y_end - y_start + 1
        w_task = x_end - x_start + 1
        rect = patches.Rectangle((x_start, y_start), 
                                 w_task - border_width / 2, h_task - border_width / 2, linewidth=border_width, edgecolor='r', facecolor='none')
        return rect
    
    desired_width_px = 1504  # Make sure this is divisible by 16
    desired_height_px = 512  # Make sure this is divisible by 16
    dpi = 100  # Adjust dpi to control the quality/size of the frame

    # Calculate figure size in inches
    fig_width = desired_width_px / dpi
    fig_height = desired_height_px / dpi
    fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    plt.subplots_adjust(hspace=0.5)
    canvas = FigureCanvas(fig)

    def update_plots(t, axs):
        for ax_row in axs:
            for ax in ax_row:
                ax.clear()

        axs[0, 0].set_title('Agent trajectory')    
        axs[0, 1].set_title('Expected current targets')

        axs[1, 0].set_title('Agent trajectory history')
        axs[1, 1].set_title('Arriving targets per time step')
        
        # Agent trajectory 
        trajectory_grid = create_trajectory_grid(loc_h_history, loc_w_history, sensor_coverages, t, L, H, W)
        pos = axs[0, 0].imshow(trajectory_grid, cmap='hot', vmin=0, vmax=2)
        task_rect = create_task_rectangle(task_areas[t])
        # Add the rectangle patch to the plot
        axs[0, 0].add_patch(task_rect)
            
        # Expected arrivals
        pos = axs[0, 1].imshow(grid_history[t], cmap='viridis', vmin=0, vmax=g_max)
        if t == 0:
            fig.colorbar(pos, ax=axs[0, 1])

        # Occupation intensities
        pos = axs[1, 0].imshow(state_history[t], cmap='viridis', vmin=0, vmax=sh_max)
        if t == 0:
            fig.colorbar(pos, ax=axs[1, 0])

        # Additions per time step 
        pos = axs[1, 1].imshow(birth_history[t], cmap='viridis', vmin=0, vmax=b_max)
        if t == 0:
            fig.colorbar(pos, ax=axs[1, 1])


    video_path = 'visualization_video.mp4'
    with imageio.get_writer(video_path, fps=N) as writer:
        for t in range(T):
            update_plots(t, axs)
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()
            frame = np.frombuffer(s, np.uint8).reshape((height, width, 4))
            writer.append_data(frame[:, :, 0:3])
#            frame_path = os.path.join(temp_dir, f'frame_{t:04d}.png')
#            frame = imageio.imread(frame_path)
#            writer.append_data(frame)

    plt.close()

    
    # Create the video



    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Rendering took {execution_time:.2f} seconds to finish")
    print(f"Earned reward: {episode_td['episode_reward'][-1].item()}")
    # Clean up the temporary directory

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()