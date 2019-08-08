"""
Purpose: Contains the function to compute the KL divergence of the 1-D distribution
Author: Vineet Madan
Date: 6 July 2019
"""

import torch
import numpy as np

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

    num_samples = real_data.size(0).item()

    num_bins = float(max_obs-min_obs)/float(bin_size)

    real_bins = torch.histc(real_data, num_bins, min_obs, max_obs)/float(num_samples)

    pred_bins = torch.histc(predicted_data, num_bins, min_obs, max_obs)/float(num_samples)

    real_entropy = torch.mul(real_bins*torch.log(real_bins), float(-1))
    cross_entropy = torch.mul(real_bins*torch.log(pred_bins), float(-1))

    return cross_entropy-real_entropy