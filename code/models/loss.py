import torch
import torch.nn.functional as F


CONST_KL = 1.0
CONST_MSE = 1.0


def loss_BCD(Y, MR, out_Mnet_mean, out_Mnet_var, Y_rec, sigmaRui_sq, theta_val=0.5):
    """
    Args:
        Y: OD image, shape: batch_size x 3 x H x W
        MR: Ruifrok matrix, shape: batch_size x 3 x 2
        out_Mnet_mean: output of Mnet, mean for the color vector matrix, shape: batch_size x 3 x 2
        out_Mnet_var: output of Mnet, variance for the color vector matrix, shape: batch_size x 1 x 2
        Y_rec: reconstructed OD image, shape: batch_size x 3 x H x W
        sigmaRui_sq: variance of the Ruifrok prior, shape: batch_size x 1 x 2
        theta_val: weight of the KL divergence term, scalar        
    """

    out_Mnet_var_div_sigmaRui_sq = torch.div(out_Mnet_var, sigmaRui_sq) # shape: (batch, 1, 2)

    # el primero es el h, y el segundo es el e
    mse_mean_vecs = F.mse_loss(out_Mnet_mean, MR, reduction='none') # shape: (batch, 3, 2)
    mse_mean_vecs = torch.sum(mse_mean_vecs, 1) # shape: (batch, 2)

    term_kl1 = 0.5 * torch.sum(mse_mean_vecs / sigmaRui_sq) # shape: (1,)
    term_kl2 = (1.5) * torch.sum(out_Mnet_var_div_sigmaRui_sq - torch.log(out_Mnet_var_div_sigmaRui_sq) - 1) # shape: (1,)
    loss_kl = term_kl1 + term_kl2 #shape: (1,)
    
    loss_mse = F.mse_loss(Y, Y_rec, reduction='sum') # shape: (1,)
    
    # print('loss mse:',loss_mse)

    loss_kl = CONST_KL*loss_kl
    loss_mse = CONST_MSE*loss_mse

    loss = (1.0-theta_val)*loss_mse + theta_val*loss_kl

    return loss, loss_kl, loss_mse

def loss_BCD_old(Y, MR, Y_rec, out_Cnet, out_Mnet_mean, out_Mnet_var, sigmaRui_sq, theta_val=0.5):
    """
    Args:
        out_CNet: output of CNet, estimation of the separated concentrations in the image, one layer for each stain,
            shape: batch x 2 x H x W
        out_MNet_mean: output of MNet, mean for the color vector matrix
            shape: batch x 3 x ns
        out_MNet_var: output of MNet, variance for the color vector matrix
            shape: batch x 1 x ns
        Y: OD image, shape: batch x 3 x H x W
        sigma_s2: 
        MR:
        patch_size:
    Returns:
        loss
    """
    
    #out_Mnet_var_div_sigmaRui_sq = torch.stack([torch.div(out_Mnet_var, sigmaRui_sq[0]), torch.div(out_Mnet_var, sigmaRui_sq[1])], 2)
    out_Mnet_var_div_sigmaRui_sq = torch.div(out_Mnet_var, sigmaRui_sq) # shape: (batch, 1, 2)

    # el primero es el h, y el segundo es el e
    mse_mean_vecs = F.mse_loss(out_Mnet_mean, MR, reduction='none') # shape: (batch, 3, 2)
    mse_mean_vecs = torch.sum(mse_mean_vecs, 1) # shape: (batch, 2)
    #mse_mean_vecs = torch.sum(torch.pow(out_Mnet_mean - MR, 2), dim=[1,2]) # shape: (batch, 2)

    #term_kl1 = 0.5 * torch.sum(mse_mean_vecs / sigmaRui_sq, dim=1) # shape: (batch,)
    #term_kl2 = (1.5) * torch.sum(out_Mnet_var_div_sigmaRui_sq - torch.log(out_Mnet_var_div_sigmaRui_sq) - 1, dim=[1,2]) # shape: (batch,)
    #loss_kl = torch.mean(term_kl1 + term_kl2) #shape: (1,)
    term_kl1 = 0.5 * torch.sum(mse_mean_vecs / sigmaRui_sq) # shape: (1,)
    term_kl2 = (1.5) * torch.sum(out_Mnet_var_div_sigmaRui_sq - torch.log(out_Mnet_var_div_sigmaRui_sq) - 1) # shape: (1,)
    loss_kl = term_kl1 + term_kl2 #shape: (1,)
    
    #term_mse1 = F.mse_loss(Y, Y_rec, reduction='mean') # shape: (1,)
    #patch_size = out_Cnet.shape[2]
    #Cflat = out_Cnet.view(-1, 2, patch_size * patch_size) #shape: (batch, 2, patch_size * patch_size)
    #Cflat_swp = Cflat.permute(0, 2, 1) #shape: (batch, patch_size * patch_size, 2)
    #term_mse2 = 3.0 * torch.mean( torch.sum(torch.mul(out_Mnet_var, torch.pow(Cflat_swp, 2)), dim=[1,2]) ) # shape: (1,)
    #term_mse2 = 0.0
    #loss_mse = term_mse1 + term_mse2
    
    #sample_M = out_Mnet_mean + torch.rand_like(out_Mnet_mean)*out_Mnet_var
    H = out_Cnet.shape[2]
    W = out_Cnet.shape[3]
    batch_size = out_Cnet.shape[0]
    Cflat = out_Cnet.view(-1, 2, H * W) #shape: (batch, 2, patch_size * patch_size)
    Y_rec_sample = Y_rec + torch.matmul(torch.rand_like(out_Mnet_mean)*torch.sqrt(out_Mnet_var), Cflat).view(batch_size, 3, H, W)
    #loss_mse = F.mse_loss(Y, Y_rec_sample, reduction='mean') # shape: (1,)
    loss_mse = F.mse_loss(Y, Y_rec_sample, reduction='sum') # shape: (1,)
    
    # print('loss mse:',loss_mse)

    loss_kl = CONST_KL*loss_kl
    loss_mse = CONST_MSE*loss_mse

    #loss_mse = 0.5 * (1.0/theta_val**2) * loss_mse
    
    #loss_mse = (1.0-theta_val)*loss_mse
    #loss_kl = (theta_val)*loss_kl
    
    #loss = loss_mse + loss_kl
    loss = (1.0-theta_val)*loss_mse + theta_val*loss_kl

    return loss, loss_kl, loss_mse
    #return {'loss': loss, 'loss_mse': loss_mse, 'loss_kl': loss_kl, 'loss_kl_h': loss_kl_h, 'loss_kl_e': loss_kl_e, }
