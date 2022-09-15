import torch
from math import pi, log
import torch.nn.functional as F

def loss_fn(out_MNet1,out_MNet2, prior_sigma_h2,prior_sigma_e2,mR):
    """
    Args:
        out_CNet: output of ConcentrationNet, parameter of Gaussian distribution q(x|k,y), first 3 channel is mean of latent clean image,
                  second part is the log-variance of latent clean image, shape: batch x 6 x H x W
        out_MNet: output of ColorNet, mode of Dirichlet distribution q(k|y), mode * weight + 1 is the parameter of Dirichlet distribution,
                  shape: batch x 1 x kernel_size x kernel_size
        y: OD image, shape: batch x channel x H x W
        alpha_h2: variance of prior p(x), Gaussian, scalar
        alpha_e2: variance of p(y|k,x), Gaussian, scalar
    Returns:
        negative ELBO
    """
    # clip bound
    log_max = log(1e4)
    log_min = log(1e-10)

    B = out_MNet1.shape[0]
    mR_e=mR[:,0,-1,:] ####[[0.0928, 0.9541, 0.2831]]  16,3
    mR_h=mR[:,0,0,:] #####[[0.6442, 0.7166, 0.2668]]

    # parameters predicted of Gaussian distribution q(vh|y)
    m_h = out_MNet1[:, 0,0,:]  # q(vh|y): m_h ##16,3

    # sigma_h2 = torch.exp(out_KNet2[:, 0].clamp(min=log_min, max=log_max))   # q(vh|y): variance 16,1
    sigma_h2 =out_MNet2[:, 0]
    # print(sigma_h2)
    # KL divergence of Gauss distribution
    sigma_h2_div_alpha_h2 = torch.div(sigma_h2, prior_sigma_h2) ######16,1
    norm_h=torch.norm(m_h - mR_h,dim=1).view(B,1)
    kl_vh_gauss = 0.5 * (torch.mean( norm_h** 2 / prior_sigma_h2 + (sigma_h2_div_alpha_h2 - 1 - torch.log(sigma_h2_div_alpha_h2))))



    # parameters predicted of Gaussian distribution q(ve|y)
    m_e = out_MNet1[:, 0,1,:]  # q(ve|y): m_e ##16,3
    # sigma_e2 = torch.exp(out_KNet2[:, 1].clamp(min=log_min, max=log_max))  # q(ve|y): variance#########16,3
    sigma_e2 =out_MNet2[:,1]

    # print(out_KNet2[:, 0, 1, :])
    # print(sigma_e2)
    # KL divergence of Gauss distribution
    sigma_e2_div_alpha_e2 = torch.div(sigma_e2, prior_sigma_e2) ####16,1
    norm_e=torch.norm(m_e - mR_e, dim=1).view(B, 1)

    kl_ve_gauss = 0.5 * (torch.mean(norm_e ** 2 / prior_sigma_e2 + (sigma_e2_div_alpha_e2 - 1 - torch.log(sigma_e2_div_alpha_e2))))



    loss =  0.5*(kl_vh_gauss + kl_ve_gauss)

    return loss, kl_vh_gauss, kl_ve_gauss


