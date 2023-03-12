import torch
import numpy as np

from torchmetrics.functional import structural_similarity_index_measure

from tqdm import tqdm

from .loss import loss_BCD
from .networks.cnet import get_cnet
from .networks.mnet import get_mnet

from utils.callbacks import CallbacksList

def peak_signal_noise_ratio(A, B, max=255.0):
    """
    input: 
        A: tensor of shape (N, C, H, W)
        B: tensor of shape (N, C, H, W)
    return:
        psnr: tensor of shape (N, )
    """
    if max is None:
        max = torch.max(A)
    mse = torch.mean((A - B)**2, dim=(1,2,3))
    psnr = 10 * torch.log10(max**2 / mse)
    return psnr

def normalize(A):
    """
    input: 
        A: tensor of shape (N, C, H, W)
    return:
        A: tensor of shape (N, C, H, W)
    """
    bs = A.shape[0]
    min = A.view(bs, -1).min(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    max = A.view(bs, -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
    A = (A - min) / (max - min)
    return A

class DVBCDModule(torch.nn.Module):
    def __init__(self, cnet_name, mnet_name) -> None:
        super().__init__()
        self.cnet_name = cnet_name
        self.mnet_name = mnet_name
        self.cnet = get_cnet(cnet_name)
        self.mnet = get_mnet(mnet_name)

        self.sft = self.cnet_name[-3:] == 'sft'
    
    def forward(self, y):
        batch_size, _, H, W = y.shape 
        out_M_mean, out_M_var = self.mnet(y) # shape: (batch_size, 3, 2), (batch_size, 1, 2)
        out_M_sample = out_M_mean + torch.rand_like(out_M_mean)*torch.sqrt(out_M_var)

        if self.sft:
            out_M_mean_fl = out_M_mean.view(batch_size, -1)
            out_M_sample_fl = out_M_sample.view(batch_size, -1)
            out_C_mean = self.cnet(y, out_M_mean_fl) # shape: (batch_size, 2, H, W)
            out_C_sample = self.cnet(y, out_M_sample_fl) # shape: (batch_size, 2, H, W)
        else:
            out_C_mean = self.cnet(y) # shape: (batch_size, 2, H, W)
            out_C_sample = self.cnet(y) # shape: (batch_size, 2, H, W)
        out_C_mean_fl = out_C_mean.view(-1, 2, H * W) #shape: (batch, 2, H * W)
        out_C_sample_fl = out_C_sample.view(-1, 2, H * W) #shape: (batch, 2, H * W)

        Y_rec_od_mean = torch.matmul(out_M_mean, out_C_mean_fl) #shape: (batch, 3, H * W)
        Y_rec_od_sample = torch.matmul(out_M_sample, out_C_sample_fl) #shape: (batch, 3, H * W)
        Y_rec_od_mean = Y_rec_od_mean.view(batch_size, 3, H, W) #shape: (batch, 3, H, W)
        Y_rec_od_sample = Y_rec_od_sample.view(batch_size, 3, H, W) #shape: (batch, 3, H, W)

        return out_M_mean, out_M_var, out_C_mean, Y_rec_od_mean, out_M_sample, out_C_sample, Y_rec_od_sample

class DVBCDModel():
    def __init__(
                self, cnet_name='unet6', mnet_name='mobilenetv3s',
                sigmaRui_sq=torch.tensor([0.05, 0.05]), theta_val=0.5, 
                lr=1e-4, lr_decay=0.1, clip_grad=np.Inf
                ):

        self.module = DVBCDModule(cnet_name, mnet_name)       

        self.loss_fn = loss_BCD

        self.sigmaRui_sq = sigmaRui_sq
        self.orig_theta_val = theta_val
        self.theta_val = theta_val
        self.pretraining_theta_val = 0.999

        self.lr = lr
        self.lr_decay = lr_decay

        self.clip_grad = clip_grad

        self.stop_training = False
        self.optim_initiated = False
        self.optimizer = None
        self.lr_scheduler = None

        self.callbacks_list = CallbacksList()
        self.device = "cpu"
        self.dp = False

    
    def set_callbacks(self, callbacks):
        self.callbacks_list.set_callbacks(callbacks)
    
    def to(self, device):
        self.device = device
        self.module = self.module.to(device)
        self.sigmaRui_sq = self.sigmaRui_sq.to(device)
        return self

    def init_optimizer(self):
        self.opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        self.lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=self.lr_decay, patience=2, verbose=True)

        self.optim_initiated = True
    
    def compute_metrics(self, out_dic, prefix="train", compute_loss=False):
        
        Y_RGB = out_dic['Y_RGB'].to(self.device)
        Y_OD = self._rgb2od(Y_RGB).to(self.device)
        MR = out_dic['MR'].to(self.device)
        Y_rec_od = out_dic['Y_rec_od'].to(self.device)
        out_C_mean = out_dic['out_C_mean'].to(self.device)
        out_M_mean = out_dic['out_M_mean'].to(self.device)
        out_M_var = out_dic['out_M_var'].to(self.device)

        metrics_dic = {}
        batch_size = Y_OD.shape[0]

        if compute_loss:
            loss, loss_kl, loss_mse = self.loss_fn(Y_OD, MR, out_M_mean, out_M_var, Y_rec_od, self.sigmaRui_sq, self.theta_val)

            #loss, loss_kl, loss_mse = self.loss_fn(Y_OD, MR, Y_rec_od, out_C_mean, out_M_mean, out_M_var, self.sigmaRui_sq, self.theta_val)
            metrics_dic = {
                f"{prefix}_loss": loss.item()/batch_size, f"{prefix}_loss_mse": loss_mse.item()/batch_size, f"{prefix}_loss_kl": loss_kl.item()/batch_size
            }

        Y_RGB_norm = torch.clamp(Y_RGB, 0.0, 255.0)
        #Y_RGB_norm = 255.0*normalize(Y_RGB)
        Y_rec_rgb = self._od2rgb(Y_rec_od)
        Y_rec_rgb_norm = torch.clamp(Y_rec_rgb, 0.0, 255.0)
        #Y_rec_rgb_norm = 255.0*normalize(Y_rec_rgb)
        
        mse_rec = ((Y_OD - Y_rec_od)**2).mean(dim=(1, 2, 3)).mean()
        #mse_rec = torch.nn.functional.mse_loss(Y_OD, Y_rec_od, reduction='none').mean(dim=(1, 2, 3)).mean()
        psnr_rec = peak_signal_noise_ratio(Y_rec_rgb_norm, Y_RGB_norm).mean()
        ssim_rec = structural_similarity_index_measure(Y_rec_rgb_norm, Y_RGB_norm)
        metrics_dic = {
            **metrics_dic,
            f"{prefix}_mse_rec": mse_rec.item(), f"{prefix}_psnr_rec": psnr_rec.item(), f"{prefix}_ssim_rec": ssim_rec.item()
        }

        if 'M_GT' in out_dic.keys():
            M_GT = out_dic['M_GT'].to(self.device)
            C_GT = self._direct_deconvolution(Y_OD, M_GT).to(self.device)

            HE_pred_OD = torch.einsum('bcs, bshw -> bschw', out_M_mean, out_C_mean)
            H_pred_OD = HE_pred_OD[:, 0, :, :]
            H_pred_RGB = self._od2rgb(H_pred_OD)
            H_pred_RGB_norm = torch.clamp(H_pred_RGB, 0.0, 255.0)
            #H_pred_RGB_norm = 255.0*normalize(H_pred_RGB)

            E_pred_OD = HE_pred_OD[:, 1, :, :]
            E_pred_RGB = self._od2rgb(E_pred_OD)
            E_pred_RGB_norm = torch.clamp(E_pred_RGB, 0.0, 255.0)
            #E_pred_RGB_norm = 255.0*normalize(E_pred_RGB)

            C_GT_OD = torch.einsum('bcs, bshw -> bschw', M_GT, C_GT)
            H_GT_OD =  C_GT_OD[:, 0, :, :]
            H_GT_RGB = self._od2rgb(H_GT_OD)
            H_GT_RGB_norm = torch.clamp(H_GT_RGB, 0.0, 255.0)
            #H_GT_RGB_norm = 255.0*normalize(H_GT_RGB)
            E_GT_OD =  C_GT_OD[:, 1, :, :]
            E_GT_RGB = self._od2rgb(E_GT_OD)
            E_GT_RGB_norm = torch.clamp(E_GT_RGB, 0.0, 255.0)
            #E_GT_RGB_norm = 255.0*normalize(E_GT_RGB)

            mse_gt_h = torch.nn.functional.mse_loss(H_pred_OD, H_GT_OD, reduction='none').mean(dim=(1, 2, 3)).mean()
            mse_gt_e = torch.nn.functional.mse_loss(E_pred_OD, E_GT_OD, reduction='none').mean(dim=(1, 2, 3)).mean()
            mse_gt = (mse_gt_h + mse_gt_e) / 2.0

            psnr_gt_h = peak_signal_noise_ratio(H_pred_RGB_norm, H_GT_RGB_norm).mean()
            psnr_gt_e = peak_signal_noise_ratio(E_pred_RGB_norm, E_GT_RGB_norm).mean()
            psnr_gt = (psnr_gt_h + psnr_gt_e) / 2.0

            ssim_gt_h = structural_similarity_index_measure(H_pred_RGB_norm, H_GT_RGB_norm)
            ssim_gt_e = structural_similarity_index_measure(E_pred_RGB_norm, E_GT_RGB_norm)
            ssim_gt = (ssim_gt_h + ssim_gt_e) / 2.0

            metrics_dic = {
                **metrics_dic,
                f"{prefix}_mse_gt_h": mse_gt_h.item(), f"{prefix}_mse_gt_e": mse_gt_e.item(), f"{prefix}_mse_gt": mse_gt.item(),
                f"{prefix}_psnr_gt_h": psnr_gt_h.item(), f"{prefix}_psnr_gt_e": psnr_gt_e.item(), f"{prefix}_psnr_gt": psnr_gt.item(),
                f"{prefix}_ssim_gt_h": ssim_gt_h.item(), f"{prefix}_ssim_gt_e": ssim_gt_e.item(), f"{prefix}_ssim_gt": ssim_gt.item()
                }
        
        return metrics_dic

    def fit(self, max_epochs, train_dataloader, val_dataloader=None, pretraining=False):
        if val_dataloader is None:
            val_dataloader = train_dataloader

        self.sigmaRui_sq = self.sigmaRui_sq.to(self.device)
        if pretraining:
            self.theta_val = self.pretraining_theta_val
        else:
            self.theta_val = self.orig_theta_val
        
        if not self.optim_initiated:
            self.init_optimizer()
        
        epoch_log_dic = {}

        self.callbacks_list.on_train_begin()
        
        for epoch in range(1, max_epochs + 1):

            self.callbacks_list.on_epoch_begin(epoch)
            
            # Trainining loop:
            self.module.train()
            train_log_dic = self.train_epoch(epoch, train_dataloader)

            # Validation loop:
            self.module.eval()
            with torch.no_grad():
                val_log_dic = self.val_epoch(epoch, val_dataloader)

            epoch_log_dic = {**train_log_dic, **val_log_dic}

            if self.lr_sch is not None:
                self.lr_sch.step(epoch_log_dic['val_loss'])
            
            self.callbacks_list.on_epoch_end(epoch, epoch_log_dic)

            if self.stop_training:
                break
        
        self.callbacks_list.on_train_end()

    def train_epoch(self, epoch, dataloader):
        return self.run_epoch(epoch, dataloader, "train")

    def val_epoch(self, epoch, dataloader):
        return self.run_epoch(epoch, dataloader, "val")
    
    def test_epoch(self, epoch, dataloader):
        return self.run_epoch(epoch, dataloader, "test")
    
    def run_epoch(self, epoch, dataloader, mode="train"):
        step_method = None
        if mode == "train":
            step_method = self.train_step 
        elif mode == "val":
            step_method = self.val_step
        else:
            step_method = self.test_step

        on_step_begin = None
        on_step_end = None
        if mode == "train":
            on_step_begin = self.callbacks_list.on_train_step_begin
            on_step_end = self.callbacks_list.on_train_step_end
        elif mode == "val":
            on_step_begin = self.callbacks_list.on_val_step_begin
            on_step_end = self.callbacks_list.on_val_step_end
        else:
            on_step_begin = self.callbacks_list.on_test_step_begin
            on_step_end = self.callbacks_list.on_test_step_end


        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        pbar.set_description(f"Epoch {epoch} - {mode}")
        metrics_dic = {}
        for batch_idx, batch in pbar:
            on_step_begin(self, batch_idx)
            partial_metrics_dic = step_method(batch, batch_idx)
            on_step_end(self)

            for k, v in partial_metrics_dic.items():
                if k not in metrics_dic:
                    metrics_dic[k] = []
                metrics_dic[k].append(v)

            if batch_idx == (len(dataloader) - 1):
                for k, v in metrics_dic.items():
                    metrics_dic[k] = np.mean(v)
                pbar.set_postfix({k : str(round(v, 4)) for k, v in metrics_dic.items()})
            else:
                pbar.set_postfix({k : str(round(v, 4)) for k, v in partial_metrics_dic.items()})
        pbar.close()
        return metrics_dic

    def train_step(self, batch, batch_idx):

        Y_RGB, MR = batch
        Y_OD = self._rgb2od(Y_RGB).to(self.device)
        MR = MR.to(self.device)

        self.opt.zero_grad()

        out_M_mean, out_M_var, out_C_mean, Y_rec_od_mean, out_M_sample, out_C_sample, Y_rec_od_sample = self.module(Y_OD) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)
        
        loss, loss_kl, loss_mse = self.loss_fn(Y_OD, MR, out_M_mean, out_M_var, Y_rec_od_sample, self.sigmaRui_sq, self.theta_val)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clip_grad)

        self.opt.step()

        loss_dic = {'train_loss' : loss.item(), 'train_loss_mse' : loss_mse.item(), 'train_loss_kl' : loss_kl.item()}
        out_dic = {'Y_RGB' : Y_RGB.detach().cpu(), 'MR' : MR.detach().cpu(), 'Y_rec_od' : Y_rec_od_mean.detach().cpu(), 'out_C_mean' : out_C_mean.detach().cpu(), 'out_M_mean' : out_M_mean.detach().cpu(), 'out_M_var' : out_M_var.detach().cpu()}
        metrics_dic = self.compute_metrics(out_dic, 'train')
        
        return {**loss_dic, **metrics_dic}

    def val_step(self, batch, batch_idx):
        return self._shared_val_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_val_step(batch, batch_idx, 'test')
    
    def _shared_val_step(self, batch, batch_idx, prefix):

        with torch.no_grad():
            out_dic = {}
            if len(batch) == 2:
                Y_RGB, MR = batch
            elif len(batch) > 2:
                Y_RGB, MR, M_GT = batch
                Y_OD = self._rgb2od(Y_RGB)
                out_dic['M_GT'] = M_GT.detach().cpu()

            Y_OD = self._rgb2od(Y_RGB).to(self.device)
            MR = MR.to(self.device)

            out_M_mean, out_M_var, out_C_mean, Y_rec_od_mean, out_M_sample, out_C_sample, Y_rec_od_sample = self.module(Y_OD) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)

            loss, loss_kl, loss_mse = self.loss_fn(Y_OD, MR, out_M_mean, out_M_var, Y_rec_od_sample, self.sigmaRui_sq, self.theta_val)

            loss_dic = {f'{prefix}_loss' : loss.item(), f'{prefix}_loss_mse' : loss_mse.item(), f'{prefix}_loss_kl' : loss_kl.item()}
            out_dic = {**out_dic, 'Y_RGB' : Y_RGB.detach().cpu(), 'MR' : MR.detach().cpu(), 'Y_rec_od' : Y_rec_od_mean.detach().cpu(), 'out_C_mean' : out_C_mean.detach().cpu(), 'out_M_mean' : out_M_mean.detach().cpu(), 'out_M_var' : out_M_var.detach().cpu()}
            
            metrics_dic = self.compute_metrics(out_dic, prefix)
            return {**loss_dic, **metrics_dic}

    def evaluate(self, test_dataloader, prefix='test'):
        self.module.eval()
        with torch.no_grad():
            metrics_dic = self.run_epoch(0, test_dataloader, mode='test')
        new_metrics_dic = {}
        for k, v in metrics_dic.items():
            new_k = k.replace('test_', f'{prefix}_')
            new_metrics_dic[new_k] = v
        return new_metrics_dic
    
    def deconvolve(self, Y_OD):
        out_M_mean, _, out_C_mean, _, _, _, _ = self.module(Y_OD)
        return out_M_mean, out_C_mean
    
    def save(self, path):
        if self.dp:
            torch.save(self.module.module.state_dict(), path)
        else:
            torch.save(self.module.state_dict(), path)
    
    def load(self, path, remove_module=False):
        weights = torch.load(path)
        if remove_module:
            weights = {k.replace('module.', '') : v for k, v in weights.items()}
        if self.dp:
            self.module.module.load_state_dict(weights)
        else:
            self.module.load_state_dict(weights)
    
    def DP(self):
        self.module = torch.nn.DataParallel(self.module)
        self.dp = True
        return self
    
    def _rgb2od(self, rgb):
        #return -torch.log(rgb / 255.0 + 1e-4)
        return -torch.log( (rgb+1) / 256.0) / np.log(256.0)
    
    def _od2rgb(self, od):
        #return (torch.exp(-od) - 1e-4) * 255.0
        return 256.0 * (torch.exp(- np.log(256.0) * od)) - 1.0
    
    def _direct_deconvolution(self, Y_OD, M):
        batch_size, c, H, W = Y_OD.shape
        Y_OD = Y_OD.view(batch_size, c, -1)
        M = M.detach().cpu()
        Y_OD = Y_OD.detach().cpu()
        C = torch.linalg.lstsq(M, Y_OD).solution
        C = C.view(batch_size, 2, H, W)
        return C