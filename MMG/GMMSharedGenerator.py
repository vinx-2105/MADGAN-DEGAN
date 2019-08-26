"""
Purpose: Contains the Shared Generator Class. Modify this class if you want to change the shared
        generator architecture. This object is only formed when no sharing is selected in the run command.
        The number of shared layers can also be tweaked here by putting the shared layers in the main module.
Author: Vineet Madan
Date: 6 July 2019
"""

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
            