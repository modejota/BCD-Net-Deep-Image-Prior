import torch
import numpy as np
import abc

from torchmetrics.functional import structural_similarity_index_measure

from tqdm import tqdm

from ..callbacks import CallbacksList

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

class AbstractDVBCDModel():
    def __init__(
                self, module,
                estimate_hyperparams=False,
                lr=1e-4, lr_decay=0.1, clip_grad=None
                ):

        self.module = module       
        self.estimate_hyperparams = estimate_hyperparams

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
        return self
    
    @abc.abstractmethod
    def get_hyperparams(self):
        pass
    
    @abc.abstractmethod
    def update_hyperparams(self, out_module):
        pass

    @abc.abstractmethod
    def forward(self, batch):
        """
        Input:
            batch: (Y_RGB, M_ref)
        Output:
            module_output_dic: a dictionary with the output of self.module
        """
        pass
    
    @abc.abstractmethod
    def loss_fn(self, module_output_dic):
        """
        Input:
            module_output_dic: output of self.forward
        Output:
            loss_output_dic: a dictionary with the output of the loss function
        """
        pass

    @abc.abstractmethod
    def deconvolve(self, Y_OD):
        """
        Input:
            Y_OD: torch.Tensor of shape (batch_size, 3, H, W)
        Output:
            M: torch.Tensor of shape (batch_size, 3, n_stains)
            C: torch.Tensor of shape (batch_size, 2, H, W)
        """
        pass

    def forward_and_loss(self, batch):
        """
        Input:
            batch: (Y_RGB, M_ref)
        Output:
            module_output_dic: a dictionary with the output of self.module
            loss_output_dic: a dictionary with the output of the loss function
        """

        module_output_dic = self.forward(batch)
        loss_output_dic = self.loss_fn(loss_output_dic, self.get_hyperparams())

        return module_output_dic, loss_output_dic

    def init_optimizer(self):
        self.opt = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        self.lr_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode='min', factor=self.lr_decay, patience=2, verbose=True)

        self.optim_initiated = True
    
    def compute_metrics(self, data, module_output_dic, prefix="train", compute_loss=False):
        """
        Input:
            data: (Y_RGB, M_ref) or (Y_RGB, M_ref, M_GT)
            module_output_dic: output of self.forward
            prefix: prefix to add to the metrics
            compute_loss: if True, compute the loss
        Output:
            metrics_dic: a dictionary with the metrics       
        """
        
        if len(data) == 2:
            Y_RGB, M_ref = data
        elif len(data) == 3:
            Y_RGB, M_ref, M_GT = data

        Y_OD = self._rgb2od(Y_RGB)
        M_key = "M_mean" if "M_mean" in module_output_dic.keys() else "M"
        M = module_output_dic[M_key]
        C_key = "C_mean" if "C_mean" in module_output_dic.keys() else "C"
        C = module_output_dic[C_key]

        C_flatten = C.view(C.shape[0], C.shape[1], -1) #shape: (batch, 3, H * W)
        Y_rec_od = torch.matmul(M, C_flatten) #shape: (batch, 3, H * W)

        metrics_dic = {}

        if compute_loss:
            out_loss = self.loss_fn(module_output_dic)
            metrics_dic = { f'{prefix}_{k}' : v for k, v in out_loss.items() }

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

        if M_GT is not None:
            C_GT = self._direct_deconvolution(Y_OD, M_GT).to(self.device)

            HE_pred_OD = torch.einsum('bcs, bshw -> bschw', M, C)
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

    def fit(self, max_epochs, train_dataloader, val_dataloader=None):
        if val_dataloader is None:
            val_dataloader = train_dataloader
        
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

        self.opt.zero_grad()

        module_output_dic, loss_output_dic = self.forward_and_loss(batch)

        loss_output_dic['loss'].backward()

        if self.estimate_hyperparams:
            self.update_hyperparams(module_output_dic)
        
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.clip_grad)

        self.opt.step()

        loss_output_dic = { f'train_{k}' : v for k, v in loss_output_dic.items() }
        module_output_dic = { k : v.detach().cpu() for k, v in module_output_dic.items() }
        metrics_dic = self.compute_metrics(batch, module_output_dic, prefix='train')
        
        return {**loss_output_dic, **metrics_dic}

    def val_step(self, batch, batch_idx):
        return self._shared_val_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_val_step(batch, batch_idx, 'test')
    
    def _shared_val_step(self, batch, batch_idx, prefix):

        with torch.no_grad():
            module_output_dic = {}
            if len(batch) == 2:
                Y_RGB, M_ref = batch
            elif len(batch) > 2:
                Y_RGB, M_ref, M_GT = batch

            out_module, out_loss = self.forward_and_loss((Y_RGB, M_ref))

            loss_output_dic = { f'train_{k}' : v for k, v in out_loss.items() }
            module_output_dic = { k : v.detach().cpu() for k, v in out_module.items() }
            metrics_dic = self.compute_metrics(batch, loss_output_dic, module_output_dic, prefix=prefix)

            return {**loss_output_dic, **metrics_dic}

    def evaluate(self, test_dataloader, prefix='test'):
        self.module.eval()
        with torch.no_grad():
            metrics_dic = self.run_epoch(0, test_dataloader, mode='test')
        new_metrics_dic = {}
        for k, v in metrics_dic.items():
            new_k = k.replace('test_', f'{prefix}_')
            new_metrics_dic[new_k] = v
        return new_metrics_dic
    
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