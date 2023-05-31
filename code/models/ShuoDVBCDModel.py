import torch
import numpy as np

from .networks.mnet import MNet
from .networks.unet import UNet
from .networks.unet_add_sft import UNetAddSFT

from AbstractDVBCDModel import AbstractDVBCDModel

from .losses import LossBCD_Shuo

def get_mnet(net_name, stain_dim=3, fc_hidden_dim=50):
    return MNet(net_name, stain_dim, fc_hidden_dim)

def get_cnet(net_name):
    if net_name[-3:] == 'sft':
        print('Using SFT in UNet')
        net_name = net_name[:-3]
        num_blocks = int(net_name[-1])
        return UNetAddSFT(in_nc=3, out_nc=2, nc=64, num_blocks=num_blocks)
    else:
        num_blocks = int(net_name[-1])
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=num_blocks)

class ShuoDVBCDModule(torch.nn.Module):
    def __init__(self, cnet_name, mnet_name) -> None:
        super().__init__()
        self.cnet_name = cnet_name
        self.mnet_name = mnet_name
        self.cnet = get_cnet(cnet_name)
        self.mnet = get_mnet(mnet_name)
    
    def forward(self, y, **kwargs):

        M_mean, M_var = self.mnet(y) # shape: (batch_size, 3, 2), (batch_size, 1, 2)
        C = self.cnet(y) # shape: (batch_size, 2, H, W)

        return {'M_mean' : M_mean, 'M_var' : M_var, 'C' : C}

class ShuoDVBCDModel(AbstractDVBCDModel):
    def __init__(
                self, cnet_name='unet6', mnet_name='mobilenetv3s', 
                sigma_sq=0.05, theta_val=0.5, 
                lr=1e-4, lr_decay=0.1, clip_grad=np.Inf
                ):
                
        super(ShuoDVBCDModel, self).__init__(None, False, lr, lr_decay, clip_grad)
        self.module = ShuoDVBCDModule(cnet_name, mnet_name)
        self.sigma_sq = sigma_sq
        self.theta_val = theta_val

        self.loss = LossBCD_Shuo(sigma_sq, theta_val)
    
    def get_hyperparams(self):
        return { 'sigma_sq': self.sigma_sq, 'theta_val': self.theta_val }

    def update_hyperparams(self, out_module):
        pass

    def forward(self, batch):
        """
        Input:
            batch: (Y_RGB, M_ref)
        Output:
            {M_mean, M_var, C}
        """
        Y_RGB = batch['Y_RGB'].to(self.device)
        Y_OD = self._rgb2od(Y_RGB)
        M_mean, M_var, C = self.module(Y_OD)

        return {'M_mean' : M_mean, 'M_var' : M_var, 'C' : C}

    def loss_fn(self, module_output_dic):
        """
        Input:
            module_output_dic: output of self.forward; {M_mean, M_var, C}
        Output:
            loss_output_dic: {loss, loss_kl, loss_mse}
        """
        
        M_mean = module_output_dic['M_mean']
        M_var = module_output_dic['M_var']
        C = module_output_dic['C']

        loss, loss_kl, loss_mse = self.loss(M_mean, M_var, C)

        return {'loss' : loss, 'loss_kl' : loss_kl, 'loss_mse' : loss_mse}

    def deconvolve(self, Y_OD):
        """
        Input:
            Y_OD: torch.Tensor of shape (batch_size, 3, H, W)
        Output:
            M: torch.Tensor of shape (batch_size, 3, n_stains)
            C: torch.Tensor of shape (batch_size, 2, H, W)
        """
        M_mean, M_var, C = self.module(Y_OD)
        return M_mean, C
