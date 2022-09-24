import torch
from math import pi, log
import torch.nn.functional as F

def loss_fn(out_CNet, out_MNet_mean, out_MNet_var, y, mR, sigmaRui_h_sq, sigmaRui_e_sq, theta=0.5, pretraining=False, pre_kl=1e2, pre_mse=1e-2):
    """
    Args:
        out_CNet: output of DeblurNet, estimation of the separated concentrations in the image, one layer for each stain,
            shape: batch x 2 x H x W
        out_MNet_mean: output of KernelNet, mean for the color vector matrix
            shape: batch x 3 x ns
        out_MNet_var: output of KernelNet, variance for the color vector matrix
            shape: batch x 3 x ns
        y: OD image, shape: batch x 3 x H x W
    Returns:
        loss
    """
    # clip bound
    log_max = log(1e4)
    log_min = log(1e-10)

    mRui_h = mR[:,0,:,0]
    mRui_e = mR[:,0, :, 1]

    mu_h = out_MNet_mean[:, :, 0]
    mu_e = out_MNet_mean[:, :, 1]  

    sigma_h_sq = out_MNet_var[:, 0, 0].clamp(min=log_min, max=log_max)
    sigma_e_sq = out_MNet_var[:, 0, 1].clamp(min=log_min, max=log_max)

    # Loss beta
    sigma_h_sq_div_sigmaRui_h_sq = torch.div(sigma_h_sq, sigmaRui_h_sq)
    sigma_e_sq_div_sigmaRui_e_sq = torch.div(sigma_e_sq, sigmaRui_e_sq)

    loss_kl_h = (0.5/sigmaRui_h_sq) * torch.mean((mu_h - mRui_h) ** 2) - (3.0/2.0) * torch.sum( sigma_h_sq_div_sigmaRui_h_sq - torch.log(sigma_h_sq_div_sigmaRui_h_sq) - 1)
    loss_kl_e = (0.5/sigmaRui_e_sq) * torch.mean((mu_e - mRui_e) ** 2) - (3.0/2.0) * torch.sum( sigma_e_sq_div_sigmaRui_e_sq - torch.log(sigma_e_sq_div_sigmaRui_e_sq) - 1)
    
    loss_kl = loss_kl_h + loss_kl_e

    # Loss MSE

    batch_size, c, heigth, width = out_CNet.shape 
    # heigth = width = patch_size
    C_tensor = out_CNet.view(batch_size, 2, heigth * width)
    od_img = torch.matmul(out_MNet_mean, C_tensor)  
    od_img = od_img.view(batch_size, 3, heigth, width) # shape: (batch, 3, patch_size * patch_size)

    sum_c_h_sq = torch.sum(out_CNet[:, 0, ]**2)
    sum_c_e_sq = torch.sum(out_CNet[:, 1, ]**2)

    loss_mse = torch.linalg.norm(y - od_img)**2 + 3*(sum_c_e_sq**2)*(sigma_e_sq) + 3*(sum_c_h_sq**2)*(sigma_h_sq)
    loss_mse = torch.sum(loss_mse)

    if pretraining:
        loss = pre_mse*loss_mse + pre_kl*loss_kl
    else:
        loss = (1-theta)*loss_mse + theta*loss_kl

    return loss, loss_mse, loss_kl, loss_kl_h, loss_kl_e


