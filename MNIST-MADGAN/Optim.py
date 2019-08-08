import torch.optim as optim


def get_optim(model_params, lr, beta1, beta2):
    return optim.Adam(model_params, lr = lr, betas = (beta1, beta2))