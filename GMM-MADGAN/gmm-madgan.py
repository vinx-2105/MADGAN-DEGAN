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



#add the command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', help='Number of epochs to run. Default=30', default=30, type =int)
parser.add_argument('--gpu', help='Use 0 for CPU and 1 for GPU. Default=1', default=1, type =int)
parser.add_argument('--noise', help='Use 0 for CPU and 1 for GPU. Default=0', default=0, type =int)
# parser.add_argument('--num_channels', help='Number of channels in the real images in the real image dataset. Default=1', default=1, type=int)
# parser.add_argument('--image_size', help='The size to which the input images will be resized. Default=32', default=32, type=int)
parser.add_argument('--leaky_slope', help='The negative slope of the Leaky ReLU activation used in the architecture. Default=0.2', default=0.2, type=float)
# parser.add_argument('--dataroot', help='The parent dir of the dir(s) that contain the data. Default=\'./data\'', default='./data', type =str),
parser.add_argument('--n_z', help='The size of the noise vector to be fed to the generator. Default=64', default=64, type=int)
parser.add_argument('--batch_size', help='The batch size to be used while training. Default=240', default=240, type=int)
parser.add_argument('--num_generators', help='Number of generators to use. Default=4', default=4, type=int)
parser.add_argument('--degan', help ='1 if want to use modified loss function otherwise 0. Default=0', default=0, type=int)
parser.add_argument('--sharing', help='1 if you want to use the shared generator. 0 otherwise. Default=0', default=0, type=int)
parser.add_argument('--gpu_add', help='Address of the GPU you want to use. Default=0', default=0, type=int)
parser.add_argument('--lrg', help='Learning rate for the generator', default=1e-4, type=float)
parser.add_argument('--lrd', help='Learning rate for the discriminator', default=1e-4, type=float)
parser.add_argument('--bt1', help='Beta 1 parameter of the Adam Optimizer. Default=0.5', default=0.5, type=float)
parser.add_argument('--bt2', help='Beta 2 parameter of the Adam Optimizer. Default=0.999', default=0.999, type=float)
parser.add_argument('--ni', help='Noise degaradation interval. Default=1000', default=1000, type=int)
parser.add_argument('--ndf', help='Noise degradation factor. Default=0.98', default=0.98, type=float)
parser.add_argument('--nd', help='Noise standard dev. Default=0.1', default=0.1, type=float)
parser.add_argument('--nm', help='Noise mean. Default=0.0', default=0.0, type=float)
parser.add_argument('--chk_interval', help='Check Interval. Default=500', default=500, type=int)


"""
The params defined by the command line args
"""
args=parser.parse_args()
print(args)

################################
num_epochs = args.epochs
is_gpu = args.gpu
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
gpu_add = args.gpu_add
lrd = args.lrd
lrg = args.lrg
beta1 = args.bt1
beta2 = args.bt2
NOISE_INTERVAL=args.ni
NOISE_DEGRADATION_FACTOR=args.ndf
NOISE_DEV=args.nd
NOISE_MEAN = args.nm
CHECK_INTERVAL = args.chk_interval



CWD = os.getcwd()

SUB_DIR = 'GMM-is_degan'+str(is_degan)+'&epc='+str(num_epochs)+'sharing'+str(is_sharing)+'lrd='+str(lrd)+'&lrg='+str(lrg)
SAVE_DIR = str(CWD)+'/'+SUB_DIR


try:
    os.mkdir(SAVE_DIR)
except:
    print("SAVE DIR Already Exists")

#Init the Logger defined in Logger.py
logger = Logger(SAVE_DIR+'/log.txt')

device = torch.device("cuda:"+str(gpu_add) if (torch.cuda.is_available() and is_gpu > 0) else "cpu")
print("DEVICE IS {}".format(device))
################################

"""
This section is for raising warnings/exceptions related to the command line args
"""
############################################################################################
if(is_gpu>1 or is_gpu<0):
    raise ValueError("gpu arg is either one or zero. You entered {}".format(is_gpu))
# if(num_channels<=0):
#     raise ValueError("num_channels has to be greater than 0. You entered {}".format(num_channels))
# if(image_size<=0):
#     raise ValueError("image_size has to be greater than 0. You entered {}".format(image_size))
if(leaky_slope>0.5):
    warnings.warn("the negative slope argument of the LeakyReLU activation is unusually low. You entered {}".format(leaky_slope))
# if(os.path.isdir(dataroot)==False):
#     raise FileNotFoundError("the path specified in dataroot is not valid. You entered {}".format(dataroot))
if(n_z<64):
    warnings.warn("The length of the noise vector is unusually low. You entered {}".format(n_z))
if(batch_size<=0):
    raise ValueError("Invalid batch size. Has to be greater than zero. You entered {}".format(batch_size))
if(num_generators<=0):
    raise ValueError("Invalid number of generators. Has to be greater than zero. You entered {}".format(num_generators))
if(is_degan<0 or is_degan>1):
    raise ValueError("degan parameter is either zero or one. You entered {}".format(is_degan))
if(is_sharing<0 or is_sharing>1):
    raise ValueError("sharing parameter is either zero or one. You entered {}".format(is_sharing))
###################################################################################################



"""
This section deals with loading the data
"""
################################
#Function which returns the dataloader
def get_dataloader():
    """
    Synthetic Dataset Preparation :- GMM Distribution ( 5 mixture component ) with modes at 10 , 20 , 60 , 80 , 110 and standard deviations : 3, 3, 2, 2 and 1, respectively 
    """

    MEANS = [10, 20, 60, 80 ,110]
    DEVS = [3,3,2,2,1]
    SAMPLES_FOR_EACH = 200000#Sample two hundred thousand samples from each dist
    train_data=[]
    for i in range(0,len(MEANS)):
        train_data.extend(np.random.normal(MEANS[i],DEVS[i],SAMPLES_FOR_EACH))


   
    train_data = torch.Tensor(train_data).view(-1,1)
    print(train_data.size())


    data = data_utils.TensorDataset(train_data)
    dataloader = data_utils.DataLoader(data,batch_size=batch_size,shuffle=True)

    return dataloader
################################


"""
Initialize the weights in this cell
"""
################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') !=-1:
        nn.init.normal_(m.weight.data, ARGS['INIT_MEAN'], ARGS['INIT_DEV'])
        nn.init.normal_(m.bias.data, ARGS['INIT_MEAN'], ARGS['INIT_DEV'])

################################

"""
Initialize the generator and the discriminator and dataloader
"""
##########################################
dataloader = get_dataloader()

if is_sharing==0:
    # generator = MNISTUnsharedGenerator(num_generators, n_z,  batch_size).to(device)
    generator = GMMUnsharedGenerator(n_z).to(device)
else:
    generator = GMMSharedGenerator(n_z).to(device)

generator.apply(weights_init)

discriminator = GMMDiscriminator(num_generators, leaky_slope).to(device)
discriminator.apply(weights_init)
##########################################

"""
Init the optimizers for the generator and the discriminator and the losses
"""
##########
loss = nn.CrossEntropyLoss()

optimD = torch.optim.Adam(discriminator.parameters(), lr = lrd, betas = (beta1, beta2))
optimG = torch.optim.Adam(generator.parameters(), lr = lrg,  betas = (beta1, beta2))
##########

"""
Create the directories for storing the results
"""
######################
if os.path.isdir(SAVE_DIR):
    warnings.warn("{} DIRECTORY ALREADY EXISTS. YOU ARE OVERWRITING EXISTING DATA".format(SAVE_DIR))
else:
    os.mkdir(SAVE_DIR)

if not os.path.isdir(SAVE_DIR+'/Results'):
        os.mkdir(SAVE_DIR+'/Results')
        os.mkdir(SAVE_DIR+'/Results/ALL/')

        for x in range(num_generators):
            os.mkdir(SAVE_DIR+'/Results/G'+str(x)+'/')


######################


"""
Init the loss lists for G and D
"""
#########
D_losses = []
G_losses = []
#########

iters=0

num_batches = len(dataloader)

DEBUG=True

fixed_noise = utils.generate_noise_for_generator(batch_size//num_generators, n_z, device)

for epoch in range(num_epochs):
    print("Iters: {} Starting Epoch - {}/{}. See log.txt for more details".format(iters, epoch, num_epochs))
    for i, data in enumerate(dataloader, 0):
        ############################################
        #Train the discriminator first
        ############################################
    
        discriminator.zero_grad()
        #1. Train D on real data
        #fetch natch of real images
        real_images_batch = data[0].to(device)
        real_b_size = real_images_batch.size(0)

        if real_b_size!=batch_size:
            continue

        #generate labels for the real batch of data...the (k+1)th element is 1...rest are zero
        D_label_real  = utils.get_labels(num_generators, -1, real_b_size, device)

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
        D_Label_Fake =[]
        for g in range(num_generators):
            D_Label_Fake.append(utils.get_labels(num_generators, g, real_b_size//num_generators, device))
        
        D_Label_Fake = torch.cat(D_Label_Fake)
        D_Labels = torch.cat([D_label_real, D_Label_Fake])

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

        G_Labels = utils.get_labels(num_generators, -1, D_Fake_Output_G.size(0),  device)

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
                    test_outputs.append(fake[x*num_generators: (x+1)*num_generators])
                
            
                #save the ALL output
                fig = plt.figure()
                plt.suptitle('All -'+str(iters))
                plt.hist(fake.numpy(), 300, density=True, range = (0,130))
                plt.ylim((0,1))
                fig.savefig(SAVE_DIR+'/Results/ALL/'+str(iters)+'.png', dpi=fig.dpi)
                plt.close()
                
                #save the outputs of the individual generators
                for x in range(num_generators):
                    fig = plt.figure()
                    plt.suptitle('G'+str(x)+'-'+str(iters))
                    plt.hist(test_outputs[x].numpy(), 300, density=True, range = (0,130))
                    plt.ylim((0,1))
                    fig.savefig(SAVE_DIR+'/Results/G'+str(x)+'/'+str(iters)+'.png', dpi=fig.dpi)
                    plt.close()

        iters = iters+1
        DEBUG=False


"""
Save the model and the params to file
"""
para_dict = {
    'args':args,
    'g_state_dict':generator.state_dict(),
    'optim_g_state_dict':optimG.state_dict(),
    'd_state_dict':discriminator.state_dict(),
    'optim_d_state_dict': optimD.state_dict(),
    'd_losses':D_losses,
    'g_losses':G_losses,
}

PTH_SAVE_PATH = SAVE_DIR+'/model_save.pth'

utils.save_model(PTH_SAVE_PATH, para_dict)

