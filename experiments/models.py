import torch
import torchvision
from torch import nn
import torch.nn.functional as F

import utils as ut

class AE_MNIST(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encode = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),
            ut.View(-1, 256),
            nn.Linear(in_features=256, out_features=dim), 
            #ut.Normalize(),
        )
        
        self.decode = nn.Sequential(
            nn.Linear(in_features=dim, out_features=256),
            nn.ELU(),
            ut.View(-1, 256, 1, 1),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encode(x)
        return {'x_': self.decode(z), 'z': z}