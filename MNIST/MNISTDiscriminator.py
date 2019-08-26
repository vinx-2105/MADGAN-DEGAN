"""
Purpose: The Discriminator Class
Author: Vineet Madan
Date: 6 July 2019
"""


import torch
import torch.nn as nn

D_channel_factor = 4

class MNISTDiscriminator(nn.Module):
    #the input image is a 32*32 b/w format
    def __init__(self, num_generators, num_channels, leaky_slope):
        self.num_generators = num_generators
        self.leaky_slope = leaky_slope
        self.num_channels = num_channels 
        super(MNISTDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            #1. The first layer after input as described in the paper 1-->4
            nn.Conv2d(in_channels = num_channels, out_channels = D_channel_factor, kernel_size = 4, stride = 2, padding = 1, bias=False),
            #nn.BatchNorm2d(D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),

            #2. The second layer 4-->8
            nn.Conv2d(in_channels  = D_channel_factor, out_channels = D_channel_factor*2, kernel_size = 4, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(2*D_channel_factor),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),#leaky_slope is usually 0.2

            #3. The third layer 8-->16
            nn.Conv2d(in_channels = 2*D_channel_factor, out_channels = 4*D_channel_factor, kernel_size = 4, stride = 2, padding =1, bias=False),
            nn.BatchNorm2d(4*D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),

        )

        self.fc1 = nn.Linear(in_features = 4*D_channel_factor*4*4, out_features=4*D_channel_factor*4, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features = 4*D_channel_factor*4, out_features=4*D_channel_factor, bias=False)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features = 4*D_channel_factor, out_features=num_generators+1, bias=False)

    #the forward function
    def forward(self, x):#x is the input passed
        x = self.main(x)
        x = x.view(-1, 4*D_channel_factor*4*4)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)

        return x