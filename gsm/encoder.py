import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(Encoder, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.Nn = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(self.h_dim, self.z_dim)
        self.logvar = nn.Linear(self.h_dim, self.z_dim)
    
    def forward(self, x):
        h = self.Nn(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar