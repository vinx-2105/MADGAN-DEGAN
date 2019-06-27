import torch

"""
The DEGAN loss functions for generator and discriminator
"""
def D_Loss(D_Fake_Output, D_Real_Output, D_label_fake, loss_func):#the loss function ... CE in this case
    D_fake_val = torch.sigmoid(D_Fake_Output[:, num_generators])
    D_true_val = torch.sigmoid(D_Real_Output[:, num_generators])
    return loss_func(D_Fake_Output[:, :num_generators], D_label_fake)- torch.mean(torch.log(D_true_val)) - torch.mean(torch.log(1-D_fake_val))

def G_Loss(D_Fake_Output, D_Real_Output, D_label_fake, loss_func):#the loss function ... CE in this case
    D_fake_val = torch.sigmoid(D_Fake_Output[:, num_generators])
    D_true_val = torch.sigmoid(D_Real_Output[:, num_generators])
    return loss_func(D_Fake_Output[:, :num_generators], D_label_fake) + torch.mean(torch.log(D_true_val)) - torch.mean(torch.log(D_fake_val))