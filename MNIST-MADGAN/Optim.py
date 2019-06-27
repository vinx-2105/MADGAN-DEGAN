import torch.optim.Adam as adam


def get_adam(model_params, lr, beta1, beta 2):
    return adam(model_params, lr = lr, betas = (beta1, beta2))