import torch

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
    return torch.randn((batch_size, n_z, 1, 1), device = device)

#saves the model to a .pth file
def save_model(file_path, para_dict):
    torch.save(para_dict, file_path)