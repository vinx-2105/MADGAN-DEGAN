import torch
import torch.nn as nn


class GMMSharedGenerator(nn.Module):
    def __init__(self, n_z):
        super(GMMSharedGenerator, self).__init__()

        self.main = nn.Sequential(
            #1. The first layer after input as described in the paper
            nn.Linear(n_z,128),
            nn.ELU(inplace=True)
        )


        self.headA = nn.Sequential(
            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #3. The third layer
            nn.Linear(128,1)
        )

        self.headB = nn.Sequential(
             
            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #3. The third layer
            nn.Linear(128,1)
        )

        self.headC = nn.Sequential(
            
            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #3. The third layer
            nn.Linear(128,1)
        )

        self.headD = nn.Sequential(
            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #2. The second layer
            nn.Linear(128,128),
            nn.ELU(inplace=True),

            #3. The third layer
            nn.Linear(128,1)
        )

    def forward(self, x):
        return torch.cat([self.headA(self.main(x)), self.headB(self.main(x)), self.headC(self.main(x)), self.headD(self.main(x))])
            