import torch
import torch.nn as nn


class GMMUnsharedGenerator(nn.Module):
    def __init__(self, n_z):
        super(GMMUnsharedGenerator, self).__init__()


        self.headA = nn.Sequential(

             #1. The first layer after input as described in the paper
            nn.Linear(n_z,128),
            nn.ELU(inplace=True),
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
             #1. The first layer after input as described in the paper
            nn.Linear(64,128),
            nn.ELU(inplace=True),
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
             #1. The first layer after input as described in the paper
            nn.Linear(64,128),
            nn.ELU(inplace=True),
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
             #1. The first layer after input as described in the paper
            nn.Linear(64,128),
            nn.ELU(inplace=True),
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
        return torch.cat([self.headA(x), self.headB(x), self.headC(x), self.headD(x)])
            