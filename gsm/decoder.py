import torch
from torch import nn
from torch.nn import functional as F

class Decoder(nn.Module):
    def __init__(self, x_dim:int, z_dim:int):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.Nn = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.Tanh(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.Tanh(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.Tanh(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.Tanh()        
        )

        self.fc = nn.Linear(self.z_dim, self.x_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        h = self.Nn(z)
        x = self.softmax(self.fc(h))
        return h, x