import torch
import torch.nn as nn
"""
    The generator as given in the Table 6 of the paper 
"""

G_channel_factor = 64
class CelebADeepUnsharedGenerator(nn.Module):
    def __init__(self,n_z, num_channels=3):
        self.n_z = n_z
        self.num_channels = num_channels
        
        super(CelebADeepUnsharedGenerator, self).__init__()

        
        self.headA = nn.Sequential(
            
            #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = n_z, out_channels = G_channel_factor*8,kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

           #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*8,kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

            #2. The second layer(shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*4, kernel_size = 4, stride =2,  padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*4),
            nn.ReLU(inplace = True),

            #3. The third layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*4, out_channels = G_channel_factor*2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*2),
            nn.ReLU(inplace = True),

            #4. The fourth layer (unshared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*2, out_channels = G_channel_factor, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = G_channel_factor, out_channels = num_channels, kernel_size = 4, stride = 2, padding=1, bias=False),
            nn.Tanh()
        )

        self.headB = nn.Sequential(
            
            #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = n_z, out_channels = G_channel_factor*8,kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

          #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*8,kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

            #2. The second layer(shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*4, kernel_size = 4, stride =2,  padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*4),
            nn.ReLU(inplace = True),

            #3. The third layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*4, out_channels = G_channel_factor*2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*2),
            nn.ReLU(inplace = True),

            #4. The fourth layer (unshared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*2, out_channels = G_channel_factor, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = G_channel_factor, out_channels = num_channels, kernel_size = 4, stride = 2, padding=1, bias=False),
            nn.Tanh()
        )

        self.headC = nn.Sequential(
            
            #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = n_z, out_channels = G_channel_factor*8,kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

            #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*8,kernel_size = 5, stride = 1, padding = 2, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

            #2. The second layer(shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*4, kernel_size = 4, stride =2,  padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*4),
            nn.ReLU(inplace = True),

            #3. The third layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*4, out_channels = G_channel_factor*2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*2),
            nn.ReLU(inplace = True),

            #4. The fourth layer (unshared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*2, out_channels = G_channel_factor, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = G_channel_factor, out_channels = num_channels, kernel_size = 4, stride = 2, padding=1, bias=False),
            nn.Tanh()
        )
    
        
    def forward(self, x):#the address of the generator i.e 0, 1 or 2 in this case
        res = torch.cat([self.headA(x), self.headB(x), self.headC(x)])
        # print(res.size())
        return res
