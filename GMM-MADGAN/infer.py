import argparse, sys, os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

import skimage.io as io
import numpy as np


from MNISTDiscriminator import MNISTDiscriminator
from  MNISTSharedGenerator import MNISTSharedGenerator
from MNISTUnsharedGenerator import MNISTUnsharedGenerator
from ThreeGenerator import ThreeGenerator

#first load the model from the pth file provided

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Relative path of the pth file', type = str)

args = parser.parse_args()

if os.ispath(args.path)==False:
    raise ValueError("Path provided is invalid")

if "sharing0" in path:
    generator = ThreeGenerator()
else:
    generator = MNISTUnsharedGenerator()

checkpoint = torch.load(path)
generator.load_state_dict(checkpoint['g_state_dict'])
generator.eval()

