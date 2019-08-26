import torch
import matplotlib.pyplot as plt
from torch.distributions import uniform
import numpy as np



def custom_log(vector):
	res = torch.empty(vector.size())
	for i in range(vector.size(0)):
		if vector[i]==0:
			res[i]=0
		else:
			res[i] = torch.log(vector[i])
	return res

def calculate_KL_divergence(real_data, predicted_data, min_obs, max_obs, bin_size=0.1):
    """
        real_data: in case of 1D-GMM this is the dataset
        predicted_data: this is the data produced by the generators
        min_obs: the lower limit of the dataset
        max_obs: the upper limit of the dataset
            data ranges from [min_obs, max_obs)
        bin_size: the size of the bin for each class
    """
    #first create the tensor of zeros of size (max_obs-min_obs)/bin_size

    num_samples = real_data.size(0)
    # print(type(num_samples))
    num_bins = int((max_obs-min_obs)/(bin_size))

    print(real_data.type(), predicted_data.type())



    real_bins = torch.histc(real_data, num_bins, min_obs, max_obs)/float(num_samples)

    pred_bins = torch.histc(predicted_data, num_bins, min_obs, max_obs).float()/float(num_samples)

    print(real_bins, pred_bins)


    # for k in range(1000):
    # 	print(k, pred_bins[k])

    real_entropy = torch.mul(torch.dot(real_bins,custom_log(real_bins)), float(-1))
    cross_entropy = torch.mul(torch.dot(real_bins,custom_log(pred_bins)), float(-1))
    
    kl = cross_entropy-real_entropy
    print(cross_entropy, real_entropy)
    print(kl)
    return kl

#returns the labels needed while training the discriminator and the generator
def get_labels(num_generators, gen_address, batch_size,  device):
    #the valid range of gen_num is from [-1,num_generators-1]
    #-1 is for generating data labels for real data
    #0 is for the first generator and 2(in this case) for the third generator

    if gen_address <-1 or gen_address>num_generators-1:
        raise ValueError("Invalid generator number {}. Should be in range [-1,{}]".format(gen_address, num_generators-1))


    if gen_address == -1:
        gen_address = num_generators

    res = torch.full((batch_size, ), gen_address, device = device, dtype=torch.long)

    return res

#returns noise to be used as input for the generator (samples standard normal dist)

def generate_noise_for_generator(batch_size, n_z, device):
    #returns noise to be input for the generator
    distribution = uniform.Uniform(torch.Tensor([-1.0]),torch.Tensor([1.0]))
    return distribution.sample(torch.Size([batch_size*n_z])).view(batch_size,n_z).to(device)

#saves the model to a .pth file
def save_model(file_path, para_dict):
    torch.save(para_dict, file_path)


def plot_loss_graph(folder, title, loss_list):
    fig = plt.figure()
    plt.suptitle(title)
    plt.plot(loss_list)
    fig.savefig(folder+'/'+title+'.png')
    plt.close()