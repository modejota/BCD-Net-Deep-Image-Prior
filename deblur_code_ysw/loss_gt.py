#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14

import torch
import torch.nn.functional as F
from math import pi, log


# clip bound
log_max = log(1e4)
log_min = log(1e-8)

def loss_t(out_DNet,out_KNet1,out_KNet2, im_gt,m_gt, alpha0,alpha_h2,alpha_e2,eps2):
    '''
    Input:
        radius: radius for guided filter in the Inverse Gamma prior
        eps2: variance of the Gaussian prior of Z
        im_gt:[16, 2,64, 64]
    '''

    B = im_gt.shape[0]
    C = im_gt.shape[1]


    # KL divergence for Gauss distribution
    mu = out_DNet[:, :C, ]  # q(x|y,k): mu B*2*64*64

    sigma2 = torch.exp(out_DNet[:, C:, ].clamp(min=log_min, max=log_max))  # q(x|y,k): variance  B*2*64*64

    # KL divergence of Gauss distribution
    # KL(q(mu1, variance1) || p(mu2, variance2)) = 0.5 * (variance1 / variance2 - log(variance1 / variance2) - 1) + 0.5 * (mu1 - mu2)^2 / variance2
    sigma2_div_alpha2 = torch.div(sigma2, alpha0) ###B,1,1089


    kl_gauss = 0.5 * torch.mean((mu - im_gt) ** 2 / eps2 + (sigma2_div_alpha2 - 1 - torch.log(sigma2_div_alpha2)))



    mR_e=m_gt[:,0,-1,:] ####[[0.0928, 0.9541, 0.2831]]
    mR_h=m_gt[:,0,0,:] #####[[0.6442, 0.7166, 0.2668]]

    # parameters predicted of Gaussian distribution q(vh|y)
    m_h = out_KNet1[:, :,0,:]  # q(vh|y): m_h ##16,3
    sigma_h2 = torch.exp(out_KNet1[:, 0,1,:].clamp(min=log_min, max=log_max))   # q(vh|y): variance
    # print(out_KNet1[:, 0,1,:])
    # print(sigma_h2)
    # KL divergence of Gauss distribution
    sigma_h2_div_alpha_h2 = torch.div(sigma_h2, alpha_h2)

    kl_vh_gauss = 0.5 * torch.mean((m_h - mR_h) ** 2 / alpha_h2 + (sigma_h2_div_alpha_h2 - 1 - torch.log(sigma_h2_div_alpha_h2)))



    # parameters predicted of Gaussian distribution q(ve|y)
    m_e = out_KNet2[:, :,0,:]  # q(ve|y): m_e ##16,3
    sigma_e2 = torch.exp(out_KNet2[:, 0,1,:].clamp(min=log_min, max=log_max))  # q(ve|y): variance#########16,3
    # print(out_KNet2[:, 0, 1, :])
    # print(sigma_e2)
    # KL divergence of Gauss distribution
    sigma_e2_div_alpha_e2 = torch.div(sigma_e2, alpha_e2) ####16,3

    kl_ve_gauss = 0.5 * torch.mean((m_e - mR_e) ** 2 / alpha_e2 + (sigma_e2_div_alpha_e2 - 1 - torch.log(sigma_e2_div_alpha_e2)))





    loss = kl_gauss+kl_vh_gauss+kl_ve_gauss

    return loss,kl_gauss,kl_vh_gauss,kl_vh_gauss

