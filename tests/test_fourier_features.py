import torch
import torch.nn as nn
import numpy as np

class FourierFeatureModule(nn.Module):
    def __init__(self, input_dim, output_dim=256, sigma=1.0):
        super(FourierFeatureModule, self).__init__()
        # Generate a random matrix B from a Gaussian distribution
        self.B = torch.randn((input_dim, output_dim)) * sigma

    def fourier_feature_transform(self, x):
        # Apply the Fourier feature transformation
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x):
        # Transform input
        x_transformed = self.fourier_feature_transform(x)
        # Pass the transformed input through the MLP
        return x_transformed

# Example usage:
input_dim = 3  # for example, (x, y, z) inputs
hidden_dim = 128
output_dim = 64  # for example, control output
sigma = 10.0  # Scale for the Gaussian projection, tune based on your task

model = FourierFeatureModule(input_dim, output_dim, sigma)
# Example input
x = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
output = model(x)
print(output)