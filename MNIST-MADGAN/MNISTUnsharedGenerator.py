import torch
import torch.nn as nn
"""
    The generator as given in the Table 6 of the paper 
"""

G_channel_factor = 1

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class MNISTUnsharedGenerator(nn.Module):
    
    def __init__(self, num_generators, n_z,  batch_size):
        self.num_generators = num_generators
        self.n_z = n_z
        self.b_size = batch_size//self.num_generators
        super(MNISTUnsharedGenerator, self).__init__()
        
        self.gens = nn.ModuleList()
        for g in range(self.num_generators):
            self.gens.append(self.get_head())
        
    #returns the sequential layer for one of the generators
    def get_head(self):
        
        head = nn.Sequential(

            nn.Linear(in_features = self.n_z, out_features = 4*4*64*G_channel_factor, bias=False),

            Reshape((self.b_size, 64, 4,4)),

            nn.ConvTranspose2d(in_channels=64*G_channel_factor, out_channels = 32*G_channel_factor, kernel_size = 4, stride = 2, padding =1, bias=False),
            nn.BatchNorm2d(32*G_channel_factor),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=32*G_channel_factor, out_channels = 16*G_channel_factor, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16*G_channel_factor),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(in_channels=16*G_channel_factor, out_channels=8*G_channel_factor, kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(8*G_channel_factor),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=8*G_channel_factor, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        return head

    #the forward function for the generator
    def forward(self, x):
        res = []
        for g in range(self.num_generators):
            res.append(self.gens[g](x))

        return torch.cat(res)
