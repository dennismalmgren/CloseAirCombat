import torch
import tensordict
from torch import distributions as D
from torch import vmap
from torch import nn
import matplotlib.pyplot as plt
import math

def main():
    #we want 200km by 400 km area.
    #this means cells are 1000 by 1000 m.
    width = 200
    height = 400

    cell_width = 1000.0
    cell_height = 1000.0
    vehicle_speed = 200 #m / sec. 
    time_scale = cell_width / vehicle_speed #takes 5 seconds to traverse a cell.
    #takes 1000 seconds to traverse the width of the grid.
    #this actually turns out to be "200 steps", i.e., the width of the grid, which is what we want, more or less.
    expected_missile_survival_time = cell_width * width / (vehicle_speed * time_scale) 
    #New ones arrive from the east. Total of 0.01
    arrivals_per_sec = 0.01 #one every 100 seconds.
    w_b = arrivals_per_sec / height * time_scale * torch.ones((width, height), dtype=torch.float32)
    w_b[:-1, :] = 0.0

    #survival
    p_s = 1 - 1 / (1.5*expected_missile_survival_time)
    #sampling time, seconds. 
    tau = torch.tensor(time_scale, dtype=torch.float32) 
    #velocity vector
    Phi_hat = torch.tensor([[-vehicle_speed], [0]], dtype=torch.float32) 
    #acceleration variance (mean acceleration is 0)
    #lets say 5%
    sigma_w_2 = (0.05*vehicle_speed)**2
    #hence..
    F_theta = torch.eye(2)
    F_theta_phi = tau * torch.eye(2, dtype=torch.float32)

    #F_theta_phi = 0.
    accel_mat = torch.tensor([[0.5*tau**2],[tau]])
    Q_theta = sigma_w_2 * (accel_mat @ accel_mat.t())

    P = torch.eye(2) * vehicle_speed
    mean_dist = F_theta_phi @ Phi_hat
    mean_dist = mean_dist.squeeze(-1)
    dist_var = F_theta_phi @ P @ F_theta_phi.t() + Q_theta
    distrib = D.MultivariateNormal(mean_dist, dist_var, validate_args=False)
    std = torch.sqrt(distrib.variance)
    cutoff_stdevs = 3
    kernel_size_x = int(math.ceil(cutoff_stdevs * std[0] / cell_width)) * 2 + 1
    kernel_size_y = int(math.ceil(cutoff_stdevs * std[1] / cell_height)) * 2 + 1
    
    # Create a grid of (x, y) coordinates at which to evaluate the kernel
    x = torch.arange(-kernel_size_x // 2 + 1, kernel_size_x // 2 + 1, dtype=torch.float32) * cell_width
    y = torch.arange(-kernel_size_y // 2 + 1, kernel_size_y // 2 + 1, dtype=torch.float32) * cell_height
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
    xy_grid = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)
    xy_grid = xy_grid.unsqueeze(-1)
    loc = torch.zeros((2, 1))
    dists = loc - F_theta @ xy_grid
    dists = dists.squeeze(-1)
    log_prob_kernel_values = distrib.log_prob(dists).view(kernel_size_x, kernel_size_y)
    prob_kernel_values = torch.exp(log_prob_kernel_values) 
    prob_kernel_values = prob_kernel_values / prob_kernel_values.sum() #normalize them, for now. 
    prob_kernel_values = prob_kernel_values * p_s
    conv_kernel = prob_kernel_values[None, None, :, :]
    conv2d_layer = nn.Conv2d(in_channels=1, out_channels=1, 
                         kernel_size=conv_kernel.shape[-1], 
                         bias=False,
                         padding=(conv_kernel.shape[-2] // 2, conv_kernel.shape[-1] // 2))
    conv2d_layer.weight.data = conv_kernel
    conv2d_layer.weight.requires_grad = False
    w_b = w_b.reshape((1, 1, width, height))
    w_b_img = w_b.reshape((width, height)).t()
    
    plt.imshow(w_b_img.numpy())
    plt.show()
    w_u = torch.zeros_like(w_b)
    w_u = w_u + w_b
    for i in range(3000):
        w_u = w_b + conv2d_layer(w_u)
        if i % 500 == 0:
            w_u_img = w_u.reshape((width, height)).t()
            plt.imshow(w_u_img.numpy())
            plt.show()
    print('done')

if __name__ == "__main__":
    main()