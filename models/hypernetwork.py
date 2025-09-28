import torch
import torch.nn as nn
from collections import OrderedDict

class HyperNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, total_params_target):
        super(HyperNet, self).__init__()
        
        self.latent_dim = latent_dim
        self.total_params_target = total_params_target

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, total_params_target)
        )

    def forward(self, z):
        return self.network(z)
