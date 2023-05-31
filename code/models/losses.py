
import torch
import abc


def sample_from_normal(mean, covar=None):
    """
    input: 
        mean: tensor of shape (batch_size, D,)
        covar: tensor of shape (batch_size, D, D)
    return:
        sample: tensor of shape (batch_size, D,)
    """
    if covar is None:
        covar = torch.eye(mean.shape[1])
    dist = torch.distributions.MultivariateNormal(mean, covar)
    sample = dist.sample()
    return sample

def sample_from_matrix_normal(mean, covar_U=None, covar_V=None):
    """
    input: 
        mean: tensor of shape (batch_size, N, D)
        covar_U: tensor of shape (batch_size, N, N)
        covar_V: tensor of shape (batch_size, D, D)
    return:
        sample: tensor of shape (batch_size, N, D)
    """
    batch_size, N, D = mean.shape
    if covar_U is None:
        covar_U = torch.eye(N)
    if covar_V is None:
        covar_V = torch.eye(D)
    mean_fl = mean.view(batch_size, -1) # shape: (batch_size, N * D)
    covar = torch.kron(covar_U, covar_V) # shape: (batch_size, N * D, N * D)
    sample_fl = sample_from_normal(mean_fl, covar) # shape: (batch_size, N * D)
    sample = sample_fl.view(batch_size, N, D) # shape: (batch_size, N, D)
    return sample


class Loss(abc.ABC):
     
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class LossBCD_Shuo(Loss):
    
    def __init__(self, theta_val=0.5, sigma_sq=0.05, M_ref=None):
        super().__init__()
        self.name = 'LossBCD_v1'
        self.theta_val = theta_val
        self.sigma_sq = sigma_sq
        if M_ref is None:
            # Ruifrok matrix
            self.M_ref = torch.tensor([
                    [0.6442, 0.0928],
                    [0.7166, 0.9541],
                    [0.2668, 0.2831]
                    ]).float()
        else:
            self.M_ref = torch.tensor(M_ref).float()
        
        self.const_kl = 1.0
        self.const_mse = 1.0

    def __call__(self, Y, M_mean, M_var, C):
        """
        Args:
            Y: OD image, shape: batch_size x 3 x H x W
            M_mean: output of Mnet, mean for the color vector matrix, shape: batch_size x 3 x 2
            M_var: output of Mnet, variance for the color vector matrix, shape: batch_size x 2 x 2
            C: output of Cnet, stain concentration matrix, shape: batch_size x n_stains x H x W
            sigma_sq: variance of the Ruifrok prior, shape: batch_size x 1 x 2
            theta_val: weight of the KL divergence term, scalar        
        """

        self.M_ref = self.M_ref.to(Y.device)

        M_sample = sample_from_matrix_normal(M_mean, covar_U=M_var) # shape: (batch, 3, 2)
        C_flat = C.view(C.shape[0], C.shape[1], -1) # shape: (batch, n_stains, H*W)
        Y_rec_sample = torch.matmul(M_sample, C_flat) #shape: (batch, 3, H * W)

        # el primero es el h, y el segundo es el e
        mse_mean_vecs = (M_mean - self.M_ref)**2 # shape: (batch, 3, 2)
        mse_mean_vecs = torch.sum(mse_mean_vecs, 1) # shape: (batch, 2)

        term_kl1 = 0.5 * torch.sum(mse_mean_vecs / self.sigma_sq) # shape: (1,)
        term_kl2 = (1.5) * torch.sum(M_var / self.sigma_sq - torch.log(M_var) - 1) # shape: (1,)
        loss_kl = term_kl1 + term_kl2 #shape: (1,)
        
        loss_mse = torch.sum((Y - Y_rec_sample)**2) # shape: (1,)
        
        # print('loss mse:',loss_mse)

        loss_kl = self.const_kl*loss_kl
        loss_mse = self.const_mse*loss_mse

        loss = (1.0-self.theta_val)*loss_mse + self.theta_val*loss_kl

        return loss, loss_kl, loss_mse



def loss_BCD(Y, MR, out_M_mean, out_M_var, out_C_mean, sigma_sq=0.05, lambda_sq=0.5, theta_val=0.5):
    """
    Args:
        Y: OD image, shape: batch_size x 3 x H x W
        MR: Ruifrok matrix, shape: batch_size x 3 x 2
        out_M_mean: output of Mnet, mean for the color vector matrix, shape: batch_size x 3 x 2
        out_M_var: output of Mnet, variance for the color vector matrix, shape: batch_size x 2 x 2
        out_C_mean: output of Cnet, mean for the stain concentration matrix, shape: batch_size x n_stains x H x W
        Y_rec_mean: reconstructed OD image, shape: batch_size x 3 x H x W
        sigma_sq: variance of the Ruifrok prior, shape: batch_size x 1 x 2
        theta_val: weight of the KL divergence term, scalar        
    """

    out_M_sample = sample_from_matrix_normal(out_M_mean, covar_U=out_M_var) # shape: (batch, 3, 2)
    #out_C_sample = out_C_mean + torch.rand_like(out_C_mean)*torch.sqrt(out_C_var) # shape: (batch, n_stains, H, W)
    #out_C_sample = out_C_mean # shape: (batch, n_stains, H, W)
    out_C_sample_flat = out_C_mean.view(out_C_mean.shape[0], out_C_mean.shape[1], -1) # shape: (batch, n_stains, H*W)
    Y_rec_sample = torch.matmul(out_M_sample, out_C_sample_flat) #shape: (batch, 3, H * W)

    out_M_var_div_sigma_sq = out_M_var / sigma_sq # shape: (batch, 1, 2)

    # el primero es el h, y el segundo es el e
    mse_mean_vecs = (out_M_mean - MR)**2 # shape: (batch, 3, 2)
    mse_mean_vecs = torch.sum(mse_mean_vecs, 1) # shape: (batch, 2)

    term_kl1 = 0.5 * torch.sum(mse_mean_vecs / sigma_sq) # shape: (1,)
    term_kl2 = (1.5) * torch.sum(out_M_var_div_sigma_sq - torch.log(out_M_var_div_sigma_sq) - 1) # shape: (1,)
    loss_kl = term_kl1 + term_kl2 #shape: (1,)
    
    loss_mse = (0.5 / lambda_sq) * torch.sum((Y - Y_rec_sample)**2) # shape: (1,)
    
    # print('loss mse:',loss_mse)

    loss_kl = CONST_KL*loss_kl
    loss_mse = CONST_MSE*loss_mse

    loss = (1.0-theta_val)*loss_mse + theta_val*loss_kl

    return loss, loss_kl, loss_mse