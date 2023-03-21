import torch
import numpy as np
from torch import optim, nn, utils, Tensor

class DepthEstimator(nn.Module):
  def __init__(self, device="cpu"):
    super().__init__()
    self.device = device
    self.depthEstimator = nn.Sequential(nn.Linear(261, 250),
                               nn.Linear(250, 250),
                               nn.Linear(250, 150),
                               nn.Linear(150, 150),
                               nn.Linear(150, 100),
                               nn.Linear(100, 100),
                               nn.Linear(100, 75),
                               nn.Linear(75, 50),
                               nn.Linear(50, 3)
                               )

  def forward(self, x):
    x = torch.tensor(x, dtype=torch.float32)
    return self.depthEstimator(x)