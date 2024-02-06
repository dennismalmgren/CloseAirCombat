import torch
from torch import nn
from gymnasium import spaces
from torchrl.data import BoundedTensorSpec
from torch.distributions import Normal, Uniform

class SurvivalProbabilityModel:
    def __init__(self):
        pass
    def step(self):
        pass
    
    def probability_of_survival_tensor(self):
        pass

class ConstantSurvivalProbabilityModel(SurvivalProbabilityModel):
    # Survival probability is constant over time
    def __init__(self, ps: torch.tensor):
        self.ps = ps

    def step(self):
        pass
    
    def probability_of_survival_tensor(self):
        return self.ps

class ExpectedTargetsModel:
    def __init__(self):
        pass

    def step(self):
        pass

    def intensity_tensor(self):
        pass

class ConstantExpectedTargetsModel(ExpectedTargetsModel):
    # Birth expectations is constant over time
    # defined a 2D tensor
    def __init__(self, intensity: torch.tensor):
        self.intensity = intensity

    def step(self):
        pass

    def intensity_tensor(self):
        return self.intensity


class CVMotionModel:
    def __init__(self):
        pass

class GridPPP:
    #For an N by M grid, expected_targets_model 
    # provides a tensor of shape (N, M) for each time step.
    # theta and the prior cover the whole grid.
    def __init__(self, 
                 expected_targets_model: ExpectedTargetsModel,
                 survival_probability_model: SurvivalProbabilityModel,
                 theta: Uniform, #maybe this needs to be a grid. we'll see.
                 phi_prior: Normal
                 ):
        self.expected_targets_model = expected_targets_model
        self.theta = theta
        self.phi_prior = phi_prior
    
    def predict(self):
        pass

    def update(self):
        pass
