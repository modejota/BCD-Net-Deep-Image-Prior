import torch
from math import pi, log
import torch.nn.functional as F

def loss_fn(out_CNet,out_MNet1,out_MNet2,y, prior_sigma_h2,prior_sigma_e2,mR,alpha,pre_kl,pre_mse,pre_training):
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
    # sigma_e2 = torch.exp(out_KNet2[:, 1].clamp(min=log_min, max=log_max))  # q(ve|y): variance#########16,1
    sigma_e2 =out_MNet2[:,1]

    # print(out_KNet2[:, 0, 1, :])
    # print(sigma_e2)
    # KL divergence of Gauss distribution
    sigma_e2_div_alpha_e2 = torch.div(sigma_e2, prior_sigma_e2) ####16,1
    norm_e=torch.norm(m_e - mR_e, dim=1).view(B, 1)

    kl_ve_gauss = 0.5 * (torch.mean(norm_e ** 2 / prior_sigma_e2 + (sigma_e2_div_alpha_e2 - 1 - torch.log(sigma_e2_div_alpha_e2))))

    # likelihood，


    p, c, w, h = out_CNet.size()
    M=out_MNet1[:, 0, :, :]

    M = M.permute(0, 2, 1)
    C = out_CNet.view(p, 2, w * w)
    od_img = torch.matmul(M, C)  # B,3,4096
    od_img = od_img.view(p, 3, w, w)

    sum_e = 3*sigma_e2 #####16
    sum_h = 3*sigma_h2
    con_e = out_CNet[:, 1:, ]  ######16,1,64,64
    con_h = out_CNet[:, :1, ]


    c_2 = torch.sum(con_e ** 2,axis=(1,2,3))  #####16

    h_2 = torch.sum(con_h ** 2,axis=(1,2,3))
    t_2 = c_2 * sum_e + h_2 * sum_h#####16

    t_1 = torch.linalg.norm(y - od_img)

    t_11 = t_1 ** 2

    neg_loglikehood = torch.sum(t_11 + t_2)

    if pre_training:
        loss = pre_mse * neg_loglikehood + pre_kl * 0.5 * (kl_vh_gauss + kl_ve_gauss)

    else:
        loss = (1-alpha)*neg_loglikehood +alpha*0.5 * (kl_vh_gauss + kl_ve_gauss)



    return loss, neg_loglikehood,kl_vh_gauss, kl_ve_gauss




















