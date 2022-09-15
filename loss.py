import torch
from math import pi, log
import torch.nn.functional as F

def loss_fn(out_CNet, out_MNet_mean,out_MNet_var, y, batch_size,patch_size,alpha_h2,alpha_e2,mR,sigma2):
    """
    Args:
        out_CNet: output of DeblurNet, estimation of the separated concentrations in the image, one layer for each stain,
            shape: batch x 2 x H x W
        out_MNet: output of KernelNet, mean and variance for the color vector matrix
                  shape: batch x 3 x ns
        im_blurry: blurry image, shape: batch x channel x H x W
        im_gt: ground truth clean image, mean of prior p(x), shape: batch x channle x H x W
        kernel_gt: ground truth kernel, mode of Dirichlet, shape: batch x kernel_size x kernel_size
        alpha2: variance of prior p(x), Gaussian, scalar
        beta2: variance of p(y|k,x), Gaussian, scalar
    Returns:
        negative ELBO
    """
    # clip bound
    log_max = log(1e4)
    log_min = log(1e-10)



    mR_h=mR[:,0,:,0] #####[[0.6442, 0.7166, 0.2668]]
    mR_e = mR[:,0, :, 1]  ####[[0.0928, 0.9541, 0.2831]]


    # KL Loss ######################################################################################################
    # parameters predicted of Gaussian distribution q(vh|y)
    m_h = out_MNet_mean[:, :,0]  # q(vh|y): m_h ##16,3
    sigma_h2 = torch.exp(out_MNet_var[:, 0,0].clamp(min=log_min, max=log_max))   # q(vh|y): variance
    sigma_h2 = sigma_h2.view(-1,1)
    # sigma_h2= sigma_h2[:,0] #Using only one value
    # KL divergence of Gauss distribution
    sigma_h2_div_alpha_h2 = torch.div(sigma_h2, alpha_h2)
    # print('Printintg')
    # print(mR_h.size(),sigma_h2.size(), sigma_h2_div_alpha_h2.size())
    kl_vh_gauss = 0.5 * torch.mean((m_h - mR_h) ** 2 / alpha_h2 + (sigma_h2_div_alpha_h2 - 1 - torch.log(sigma_h2_div_alpha_h2)))
    # parameters predicted of Gaussian distribution q(ve|y)
    m_e = out_MNet_mean[:, :,1] # q(ve|y): m_e ##16,3
    sigma_e2 = torch.exp(out_MNet_var[:, 0,1].clamp(min=log_min, max=log_max))  # q(ve|y): variance#########16,3
    sigma_e2= sigma_e2.view(-1,1)
    # sigma_e2 = sigma_e2[0] #Using only one value
    # KL divergence of Gauss distribution
    sigma_e2_div_alpha_e2 = torch.div(sigma_e2, alpha_e2) ####16,3
    kl_ve_gauss = 0.5 * torch.mean((m_e - mR_e) ** 2 / alpha_e2 + (sigma_e2_div_alpha_e2 - 1 - torch.log(sigma_e2_div_alpha_e2)))
    kl_loss = (kl_vh_gauss + kl_ve_gauss)/2

    # OD ||Y-MC||

    C = out_CNet.view(batch_size, 2, patch_size * patch_size)
    od_img = torch.matmul(out_MNet_mean, C)  # B,3,4096
    od_img = od_img.view(batch_size, 3, patch_size, patch_size)

    mse = F.mse_loss(y, od_img)

    loss =  mse + kl_loss

    return loss, mse, kl_loss, kl_vh_gauss, kl_ve_gauss


