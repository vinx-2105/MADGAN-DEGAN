"""
Purpose: Resume training from a checkpoint of the 1D GMM experiment using the .pth file
Author: Vineet Madan
Date: 6 July 2019
"""

import argparse, sys
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.distributions as dist
import warnings
import  torch.utils.data as data_utils

import skimage.io as io
import numpy as np

from  GMMDiscriminator import  GMMDiscriminator
from GMMSharedGenerator import GMMSharedGenerator
from GMMUnsharedGenerator import GMMUnsharedGenerator


import utils
from Logger import Logger
import Losses

from datetime import date

from gmm_madgan_params import ARGS #import the paramters file


from Logger import Logger

torch.set_printoptions(precision=10)

parser = argparse.ArgumentParser()

parser.add_argument('--folder', help='directory name of the experiment', type=str)
parser.add_argument('--gpu', help='1 if run on gpu and 0 otherwise', type=int)
parser.add_argument('--gpu_add', help='address of the gpu', type=int)
parser.add_argument('--more_epochs', help='number of more epochs to run', type=int)


args = parser.parse_args()
print(args)
folder = args.folder
SAVE_DIR = folder
more_epochs = args.more_epochs

logger = Logger(folder+'/log.txt', mode='a')

logger.log("---------------------------------------------------")

device = torch.device("cuda:"+str(args.gpu_add) if (torch.cuda.is_available() and args.gpu > 0) else "cpu")
print("DEVICE is {}".format(device))


"""
Retrieve everything stored in the pth file
"""
#################################
checkpoint = torch.load(args.folder+'/model_save.pth')

G_losses = checkpoint['g_losses']
D_losses = checkpoint['d_losses']
train_data = checkpoint['train_data']#this is the real dataset
g_state_dict = checkpoint['g_state_dict']
d_state_dict = checkpoint['d_state_dict']
optim_g_state_dict= checkpoint['optim_g_state_dict']
optim_d_state_dict= checkpoint['optim_d_state_dict']
args = checkpoint['args']
iters = checkpoint['iters']
curr_epoch = checkpoint['epoch']
D_Labels = checkpoint['D_Labels']
G_Labels = checkpoint['G_Labels']
D_Label_Fake = checkpoint['D_Label_Fake']
fixed_noise = checkpoint['fixed_noise']
MEANS= checkpoint['MEANS']
DEVS=checkpoint['DEVS']
num_epochs = args.epochs

add_noise = args.noise
# num_channels = args.num_channels
# image_size = args.image_size
leaky_slope = args.leaky_slope
# dataroot = args.dataroot
n_z = args.n_z
batch_size = args.batch_size
num_generators = args.num_generators
is_degan = args.degan
is_sharing = args.sharing
lrd = args.lrd
lrg = args.lrg
beta1 = args.bt1
beta2 = args.bt2
NOISE_INTERVAL=args.ni
NOISE_DEGRADATION_FACTOR=args.ndf
NOISE_DEV=args.nd
NOISE_MEAN = args.nm
CHECK_INTERVAL = args.chk_interval


###################################
print(train_data.size())
dataloader = data_utils.DataLoader(train_data,batch_size=batch_size,shuffle=True)
print(len(dataloader))
print(dataloader.batch_size)
logger.log("Resuming training from checkpoint epoch {} for {} more_epochs".format(curr_epoch, more_epochs))


if args.sharing==1:
    generator = GMMSharedGenerator(n_z)

else:
    generator = GMMUnsharedGenerator(n_z)

discriminator = GMMDiscriminator(num_generators, leaky_slope)


"""
Save the model and the params to file
"""




#init the loss function
loss = nn.CrossEntropyLoss()


#load the model weights
generator.load_state_dict(g_state_dict)
discriminator.load_state_dict(d_state_dict)

#init the optimisers
optimD = torch.optim.Adam(discriminator.parameters(), lr = lrd, betas = (beta1, beta2))
optimG = torch.optim.Adam(generator.parameters(), lr = lrg,  betas = (beta1, beta2))

optimD.load_state_dict(optim_d_state_dict)
optimG.load_state_dict(optim_g_state_dict)

generator.to(device)
discriminator.to(device)

generator.train()
discriminator.train()


num_batches = len(dataloader)
print(dataloader.batch_size)
print(num_batches)

DEBUG=True

fixed_noise = utils.generate_noise_for_generator(12000, n_z, device)

colors = ['aqua', 'orange', 'fuchsia', 'yellowgreen']


def save_checkpoint(curr_epoch):
    para_dict = {
        'epoch':curr_epoch,
        'args':args,
        'train_data':train_data,
        'g_state_dict':generator.state_dict(),
        'optim_g_state_dict':optimG.state_dict(),
        'd_state_dict':discriminator.state_dict(),
        'optim_d_state_dict': optimD.state_dict(),
        'd_losses':D_losses,
        'g_losses':G_losses,
        'iters':iters,
        'D_Label':D_Labels,
        'G_Labels':G_Labels,
    }

    PTH_SAVE_PATH = SAVE_DIR+'/model_save.pth'

    utils.save_model(PTH_SAVE_PATH, para_dict)

#begin training

for epoch in range(curr_epoch, curr_epoch+more_epochs ):
    print("Iters: {} Starting Epoch - {}/{}. See log.txt for more details".format(iters, epoch, curr_epoch+more_epochs))
    for i, data in enumerate(dataloader, 0):
        ############################################
        #Train the discriminator first
        ############################################
    
        discriminator.zero_grad()
        
        #1. Train D on real data
        #fetch batch of real images
        # print(data[0].size())
        if DEBUG:
            print("hello")
            print(dataloader.batch_size)
            print(data[0].size())
            DEBUG=False
        real_images_batch = data[0].to(device)
        real_b_size = real_images_batch.size(0)

        if real_b_size!=batch_size:
            continue

        #generate labels for the real batch of data...the (k+1)th element is 1...rest are zero
        

        #forward pass for the real batch of data and then resize  
        
        gen_input_noise = utils.generate_noise_for_generator(real_b_size//num_generators, n_z, device)
        gen_output = generator(gen_input_noise)#, real_b_size//num_generators)
        
        gen_out_d_in = gen_output.detach()
        ##############################################################
        norm = dist.Normal(torch.tensor([NOISE_MEAN]), torch.tensor([NOISE_DEV]))

        if add_noise==1:
            x_noise = norm.sample(gen_out_d_in.size()).view(gen_out_d_in.size()).to(device)
            gen_out_d_in = gen_out_d_in + x_noise 
        #################################################################

        if DEBUG: logger.log(str(D_Labels))
        
        D_output_real = discriminator(real_images_batch).view((real_b_size,-1))
        D_Fake_Output = discriminator(gen_out_d_in).view((real_b_size, -1))

        D_Output = torch.cat([D_output_real, D_Fake_Output])
        
        if iters%NOISE_INTERVAL==0:
            NOISE_DEV=NOISE_DEV*NOISE_DEGRADATION_FACTOR
            logger.log("NOISE DEV IS NOW :{}".format(NOISE_DEV))

       
        if is_degan==1:
            err_D = Losses.D_Loss(D_Fake_Output, D_output_real, D_Label_Fake, loss, num_generators)
        else:
            err_D = loss(D_Output, D_Labels)


        err_D.backward(retain_graph=True)

        optimD.step()

        ########################################
        #Train the generators
        ########################################

        generator.zero_grad()

        if add_noise==1:
            D_Fake_Output_G = discriminator(gen_output+x_noise).view((real_b_size, -1))
        else:
            D_Fake_Output_G = discriminator(gen_output).view((real_b_size, -1))


        if is_degan==1:
            err_G = Losses.G_Loss(D_Fake_Output_G, D_output_real, D_Label_Fake, loss, num_generators)
        else:
            err_G = loss(D_Fake_Output_G, G_Labels)


        err_G.backward()

        optimG.step()


        if iters%CHECK_INTERVAL==0:
            logger.log("Iters: {}; Epo: {}/{}; Btch: {}/{}; D_Err: {}; G_Err: {};".format(iters, epoch, num_epochs, i,num_batches,  err_D.item(), err_G.item()))


        #add to the dicts for keeping track of losses
        D_losses.append(err_D.item())
        G_losses.append(err_G.item())


        if iters%CHECK_INTERVAL==0 or (epoch==num_epochs-1 and i==num_batches-1):
            print("Iters:{}; Epo: {}/{}; Btch: {}/{}; D_Err: {}; G_Err: {}; ".format(iters, epoch, num_epochs, i,num_batches,  err_D.item(), err_G.item()))
            samples_for_test = fixed_noise

            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu().view(-1)
                obs_size = fake.size(0)
                obs_size=obs_size//num_generators

                test_outputs = []#ith element of list stores the output of the ith generator

                for x in range(num_generators):
                    test_outputs.append(fake[x*obs_size: (x+1)*obs_size])

                #save the ALL output
                fig = plt.figure()
                plt.suptitle('All -'+str(iters))
                plt.hist(fake.numpy(), 300, density=True, range = (0,130))
                plt.ylim((0,0.5))
                fig.savefig(SAVE_DIR+'/Results/ALL/'+str(iters)+'.png', dpi=fig.dpi)
                plt.close()

                #colored output for all graph
                fig = plt.figure()
                if is_degan==1:
                    plt.suptitle('DEGAN -'+str(iters))
                else:
                    plt.suptitle('MADGAN -'+str(iters))
                plt.xticks(MEANS)
                plt.ylim((0,0.4))
                for x in range(num_generators):
                    plt.hist(test_outputs[x].numpy(), 300, density=True, range = (0,140), color=colors[x], label=colors[x]+"=G"+str(x))
                plt.legend()
                fig.savefig(SAVE_DIR+'/Results/COLOR/'+str(iters)+'.png', dpi=fig.dpi)
                plt.close()


                #save the outputs of the individual generators
                for x in range(num_generators):
                    fig = plt.figure()
                    plt.suptitle('G'+str(x)+'-'+str(iters))
                    plt.hist(test_outputs[x].numpy(), 300, density=True, range = (0,130), color=colors[x])
                    plt.ylim((0,0.5))
                    fig.savefig(SAVE_DIR+'/Results/G'+str(x)+'/'+str(iters)+'.png', dpi=fig.dpi)
                    plt.close()

        iters = iters+1
        DEBUG=False
    #save model checkpoint after every epoch
    save_checkpoint(epoch)


"""
Save the model and the params to file
"""


