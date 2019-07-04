import torch
import torch.nn as nn
"""
  The Discriminator model as given in the Table 5 of the  MADGAN paper
"""

D_channel_factor = 128

class GMMDiscriminator(nn.Module):
    
    def __init__(self, num_generators, leaky_slope):
        
        super(GMMDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            #1. The first layer after input as described in the paper
            nn.Linear(1,128),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),

            #2. The second layer
            nn.Linear(128,128),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),
            
            #2. The second layer
            nn.Linear(128,128),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),

            #3. The third layer
            nn.Linear(128,num_generators+1),
        )
    
    
    
    #the forward function
    def forward(self, x):#x is the input passed      
        return self.main(x)