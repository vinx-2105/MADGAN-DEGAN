import torch
import torch.nn as nn
"""
  The Discriminator model as given in the Table 7 of the  MADGAN paper

  I have adapted the stacked-MNIST architecture for simple MNIST 
"""

"""
  The Residual Discriminator model as given in the Table 11 of the paper
"""

D_channel_factor=64

class ResidualDiscriminator(nn.Module):
    #the input image is a 64*64 color format
    def __init__(self,  num_channels, leaky_slope, num_generators):
        super(ResidualDiscriminator, self).__init__()
        self.num_channels = num_channels
        self.leaky_slope = leaky_slope
        self.num_generators = num_generators
        self.main = nn.Sequential(
            #1. The first layer 3*3 conv. 64 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels = num_channels, out_channels = D_channel_factor, kernel_size = 7, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),
            
            #2. The second layer 3*3 conv. 64 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels  = D_channel_factor, out_channels = D_channel_factor, kernel_size = 3, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(D_channel_factor),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),#leaky_slope is usually 0.2
            
            #3. The third layer 3*3 conv. 128 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels  = D_channel_factor, out_channels = D_channel_factor*2, kernel_size = 3, stride = 2, padding = 1, bias=False),
            nn.BatchNorm2d(2*D_channel_factor),
            nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),#leaky_slope is usually 0.2
            
            
            #4. The fourth layer 3*3 conv. 256 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels = 2*D_channel_factor, out_channels = 4*D_channel_factor, kernel_size = 4, stride = 2, padding =1, bias=False),
            nn.BatchNorm2d(4*D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),
            
            #5. The fifth layer 3*3 conv. 512 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels = 4*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size = 3, stride = 1, padding =1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),
            
            #6. The sixth layer 3*3 conv. 512 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size = 3, stride = 1, padding =1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),
            
            #7. The seventh layer 3*3 conv. 512 LeakyReLU stride 2 padding 1 batchnorm
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size = 3, stride = 1, padding =1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
            nn.LeakyReLU(negative_slope = leaky_slope, inplace=True),
    

        )
        
        self.res11 = nn.Sequential(
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
            nn.ReLU()
        )
        
        self.res12 = nn.Sequential(
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
        )
        
        self.res21 = nn.Sequential(
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
            nn.ReLU()
        )
        
        self.res22 = nn.Sequential(
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
        )
        
        self.res31 = nn.Sequential(
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
            nn.ReLU()
        )
        
        self.res32 = nn.Sequential(
            nn.Conv2d(in_channels = 8*D_channel_factor, out_channels = 8*D_channel_factor, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8*D_channel_factor),
        )
        
        self.last = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=8*D_channel_factor, out_channels = 1+num_generators, kernel_size=4, stride=1, padding=0, bias=False)
        )
    
    
    #the forward function
    def forward(self, x):#x is the input passed      
        out_main = self.main(x)
        
        output11 = self.res11(out_main)
        output12 = self.res12(output11)
        out_1 = out_main+output12
        
        output21 = self.res21(out_1)
        output22 = self.res22(output21)
        out_2 = out_1 + output22
        
        output31 = self.res31(out_2)
        output32 = self.res32(output31)
        out_3 = out_2 + output32
        
        return self.last(out_3)
