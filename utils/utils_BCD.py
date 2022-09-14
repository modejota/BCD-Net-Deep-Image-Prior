import torch
import numpy as np

def torch_rgb2od(rgb):
    od = (-torch.log((rgb + 1) / 256))
    return od

def np_rgb2od(rgb):
    od = (-np.log((rgb.astype(float) + 1) / 256))
    return od

def normalize_to1(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)