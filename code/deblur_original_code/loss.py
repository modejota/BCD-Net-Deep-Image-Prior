import torch
from math import pi, log
import torch.nn.functional as F

def loss_fn(out_DNet, out_KNet, im_blurry, im_gt, kernel_gt, kernel_sample_return, alpha2, beta2, dirichlet_para_stretch):
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

    im_channel = im_gt.shape[1]  # im_gt: batch x channel x H x W
    batch_size, _, kernel_size, _ = out_KNet.size() # batch x 1 x k x k

    # parameters predicted of Gaussian distribution q(x|k,y)
    mu = out_DNet[:, :im_channel,]  # q(x|y,k): mu
    sigma2 = torch.exp(out_DNet[:, im_channel:,].clamp(min=log_min, max=log_max))   # q(x|y,k): variance
    print(out_DNet[:, im_channel:,])
    print(sigma2)
    # KL divergence of Gauss distribution
    # KL(q(mu1, variance1) || p(mu2, variance2)) = 0.5 * (variance1 / variance2 - log(variance1 / variance2) - 1) + 0.5 * (mu1 - mu2)^2 / variance2
    sigma2_div_alpha2 = torch.div(sigma2, alpha2)
    print(sigma2_div_alpha2)
    kl_gauss = 0.5 * torch.mean((mu - im_gt) ** 2 / alpha2 + (sigma2_div_alpha2 - 1 - torch.log(sigma2_div_alpha2)))

    # KL divergence of Dirichlet distribution
    # zeta: parameters of predicted Dirichlet distribution
    # zeta = out_KNet.reshape(batch_size, -1)  # batch x (kernel_size * kernel_size)
    # kernel_gt = kernel_gt.reshape(batch_size, -1)
    # p_k = torch.distributions.dirichlet.Dirichlet(kernel_gt * dirichlet_para_stretch + 1e-7)
    # q_k = torch.distributions.dirichlet.Dirichlet(zeta * dirichlet_para_stretch + 1e-7)
    # kl_dirichlet = torch.mean(torch.distributions.kl.kl_divergence(q_k, p_k))
    zeta = out_KNet.reshape(batch_size, -1) * dirichlet_para_stretch + 1e-7  # batch x (kernel_size * kernel_size)
    kernel_gt = kernel_gt.reshape(batch_size, -1) * dirichlet_para_stretch + 1e-7
    sum_p_concentration = kernel_gt.sum(-1)
    sum_q_concentration = zeta.sum(-1)
    t1 = sum_q_concentration.lgamma() - sum_p_concentration.lgamma()
    t2 = (zeta.lgamma() - kernel_gt.lgamma()).sum(-1)
    t3 = zeta - kernel_gt
    t4 = zeta.digamma() - sum_q_concentration.digamma().unsqueeze(-1)
    kl_dirichlet = torch.mean(t1 - t2 + (t3 * t4).sum(-1))


    # likelihood，根据采样出的k和x计算p(y|k,x)，其中需要采样q(k|y)和q(x|k,y)
    sample_k = kernel_sample_return # B x 1 x K x K
    sample_x = mu + torch.randn_like(mu) * torch.sqrt(sigma2) # batch x channel x H x W
    num_pad = (sample_k.size()[-1] - 1) // 2
    sample_x_pad = F.pad(sample_x, (num_pad,) * 4, mode='reflect')
    k_conv_x = F.conv3d(sample_x_pad.unsqueeze(0), sample_k.unsqueeze(1), groups=batch_size).squeeze(0)
    neg_loglikehood = 0.5 * log(2 * pi * beta2) + torch.mean(0.5 * (k_conv_x - im_blurry) ** 2 / beta2)

    loss = neg_loglikehood + kl_gauss + kl_dirichlet

    return loss, neg_loglikehood, kl_gauss, kl_dirichlet


