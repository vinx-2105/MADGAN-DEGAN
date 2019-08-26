import torch
import torch.nn as nn


G_channel_factor = 64

"""
  The generator given in table 10 of the paper
  
  For shared weights, referred to this code 
  https://discuss.pytorch.org/t/how-to-create-model-with-sharing-weight/398
"""

class CelebASharedGenerator(nn.Module):
    def __init__(self, n_z, num_channels=3):
        self.n_z = n_z
        self.num_channels = num_channels

        super(CelebASharedGenerator, self).__init__()

        self.base = nn.Sequential(
            #input is of the length n_z
            #1. First layer (shared)
            nn.ConvTranspose2d(in_channels = n_z, out_channels = G_channel_factor*8,kernel_size = 4, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(G_channel_factor*8),
            nn.ReLU(inplace = True),

            #2. The second layer(shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*8, out_channels = G_channel_factor*4, kernel_size = 4, stride =2,  padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*4),
            nn.ReLU(inplace = True),


        )

        self.headA = nn.Sequential(
            

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
           
            #3. The third layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*4, out_channels = G_channel_factor*2, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor*2),
            nn.ReLU(inplace = True),

            #4. The fourth layer (shared)
            nn.ConvTranspose2d(in_channels = G_channel_factor*2, out_channels = G_channel_factor, kernel_size = 4, stride = 2, padding = 1, bias = False),
            nn.BatchNorm2d(G_channel_factor),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(in_channels = G_channel_factor, out_channels = num_channels, kernel_size = 4, stride = 2, padding=1, bias=False),
            nn.Tanh()
        )
    
        
    def forward(self, x):#the address of the generator i.e 0, 1 or 2 in this case
        return torch.cat([self.headA(self.base(x)), self.headB(self.base(x)), self.headC(self.base(x))])

# class MNISTSharedGenerator(nn.Module):
    
#     def __init__(self, num_generators, n_z,  batch_size):
#         self.num_generators = num_generators
#         self.n_z = n_z
#         self.b_size = batch_size//self.num_generators
       
#         super(MNISTSharedGenerator, self).__init__()
        
        
#         self.fc1 = nn.Linear(in_features = self.n_z, out_features = 4*4*64*G_channel_factor)
#         self.relu1 = nn.ReLU()
        
#         self.main = nn.Sequential(

#             nn.ConvTranspose2d(in_channels=64*G_channel_factor, out_channels = 32*G_channel_factor, kernel_size = 4, stride = 2, padding =1, bias=False),
#             nn.BatchNorm2d(32*G_channel_factor),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(in_channels=32*G_channel_factor, out_channels = 16*G_channel_factor, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(16*G_channel_factor),
#             nn.ReLU(inplace=True),
#         )

#         self.heads = []

#         for g in range(self.num_generators):
#             self.heads.append[self.get_head()]


#     #returns a head sequential layer for use in this class itself
#     def get_head(self):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels=16*G_channel_factor, out_channels=8*G_channel_factor, kernel_size=4, padding=1, stride=2, bias=False),
#             nn.BatchNorm2d(8*G_channel_factor),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(in_channels=8*G_channel_factor, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.Tanh()
#         )

#     #returns the concatenation of the various head modules after passing thru main and init fc layers
#     def forward(self, x):
#         res = []
#         for g in range(1, self.num_generators):
#             res.append(self.head[g](self.main((self.relu1((self.fc1(x).view(self.b_size,64, 4,4)))))))
#         return torch.cat(res)

