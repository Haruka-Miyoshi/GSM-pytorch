import torch
from torch import nn
from torch.nn import functional as F

from .encoder import Encoder
from .decoder import Decoder

class Model(nn.Module):
    def __init__(self, x_dim:int, h_dim:int, z_dim:int):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim

        self.encode = Encoder(self.x_dim, self.h_dim, self.z_dim)
        self.decode = Decoder(self.x_dim, self.z_dim)
    
    def reparameterize(self, mu, logvar, mode):
        if mode:
            s = torch.exp(0.5 * logvar)
            e = torch.rand_like(s)
            return e.mul(s).add_(mu)
        else:
            return mu
    
    def forward(self, x, mode=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, mode)
        h, xh = self.decode(z)
        return mu, logvar, z, h, xh