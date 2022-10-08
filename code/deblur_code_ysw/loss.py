
import torch
from math import pi, log
import torch.nn.functional as F

def loss_fn(out_KNet1,out_KNet2,  alpha_h2,alpha_e2,mR):
    """
    Args:
        out_DNet: output of DeblurNet, parameter of Gaussian distribution q(x|k,y), first 3 channel is mean of latent clean image,
                  second part is the log-variance of latent clean image, shape: batch x 6 x H x W
        out_KNet: output of KernelNet, mode of Dirichlet distribution q(k|y), mode * weight + 1 is the parameter of Dirichlet distribution,
                  shape: batch x 1 x kernel_size x kernel_size
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
    al = 1e-6

    mR_e=mR[:,0,-1,:] ####[[0.0928, 0.9541, 0.2831]]  16,3
    mR_h=mR[:,0,0,:] #####[[0.6442, 0.7166, 0.2668]]

    # parameters predicted of Gaussian distribution q(vh|y)
    m_h = out_KNet1[:, 0,0,:]  # q(vh|y): m_h ##16,3

    sigma_h2 = out_KNet2[:, 0]
    # sigma_h2 = torch.exp(out_KNet1[:, 0,1,:].clamp(min=log_min, max=log_max))   # q(vh|y): variance 16,3

    # KL divergence of Gauss distribution
    sigma_h2_div_alpha_h2 = torch.div(sigma_h2, alpha_h2) #####16
    sigma_h2_div_alpha_h2=sigma_h2_div_alpha_h2.repeat(3,1).transpose(0,1)####16,3

    kl_vh_gauss = 0.5 * torch.mean((m_h - mR_h) ** 2 / alpha_h2 + (sigma_h2_div_alpha_h2 - 1 - torch.log(sigma_h2_div_alpha_h2)))



    # parameters predicted of Gaussian distribution q(ve|y)
    m_e = out_KNet1[:, 0,1,:]  # q(ve|y): m_e ##16,3
    sigma_e2 = out_KNet2[:, 1]
    # sigma_e2 = torch.exp(out_KNet2[:, 0,1,:].clamp(min=log_min, max=log_max))  # q(ve|y): variance#########16,3

    # KL divergence of Gauss distribution
    sigma_e2_div_alpha_e2 = torch.div(sigma_e2, alpha_e2) ####16
    sigma_e2_div_alpha_e2 = sigma_e2_div_alpha_e2.repeat(3, 1).transpose(0, 1)####16,3
    kl_ve_gauss = 0.5 * torch.mean((m_e - mR_e) ** 2 / alpha_e2 + (sigma_e2_div_alpha_e2 - 1 - torch.log(sigma_e2_div_alpha_e2)))






    loss = 0.5*(kl_vh_gauss + kl_ve_gauss)

    return loss,kl_vh_gauss, kl_ve_gauss