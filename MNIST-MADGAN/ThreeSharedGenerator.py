import torch
import torch.nn as nn
"""
    The generator as given in the Table 6 of the paper 
"""

G_channel_factor = 1

class ThreeSharedGenerator(nn.Module):
    
    def __init__(self, batch_size, num_generators, n_z):
        # self.num_gpu = num_gpu
        self.b_size = batch_size//num_generators
        super(ThreeSharedGenerator, self).__init__()
        
        self.fcA = nn.Linear(in_features = n_z, out_features = 4*4*64*G_channel_factor, bias=False)
        

        self.headA = nn.Sequential(
            
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

        self.headB = nn.Sequential(
            
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

        self.headC = nn.Sequential(
            
            
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
    
    

    def forward(self, x):#the address of the generator i.e 0, 1 or 2 in this case
        return torch.cat([self.headA(self.fcA(x).view(self.b_size, 64, 4,4)), self.headB(self.fcA(x).view(self.b_size, 64, 4,4)), self.headC(self.fcA(x).view(self.b_size, 64, 4,4))])
        
    