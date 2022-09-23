import torch
from math import pi, log
import torch.nn.functional as F

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


def loss_BCD(out_Cnet, out_Mnet_mean, out_Mnet_var, Y, sigma_s2, MR, patch_size):
    term_kl1 = (torch.norm(out_Mnet_mean - MR, dim=1) ** 2) / (2 * sigma_s2)
    term_kl2 = (3 / 2) * (out_Mnet_var / sigma_s2 - torch.log(out_Mnet_var / sigma_s2) - 1)
    loss_kl = torch.sum(term_kl1 + term_kl2)
    # print('loss kl:',loss_kl)

    Cflat = out_Cnet.view(-1, 2, patch_size * patch_size)
    Y_rec = torch.matmul(out_Mnet_mean, Cflat)
    term_mse1 = torch.sum(torch.norm(Y.view(-1, 3, patch_size * patch_size) - Y_rec, dim=1) ** 2)
    term_mse2 = torch.sum(torch.matmul(3 * out_Mnet_var[0], Cflat[0] ** 2))
    loss_mse = term_mse1 + term_mse2
    # print('loss mse:',loss_mse)

    loss = loss_mse + loss_kl

    return loss, loss_kl, loss_mse

    # return loss, mse, kl_loss, kl_vh_gauss, kl_ve_gauss


