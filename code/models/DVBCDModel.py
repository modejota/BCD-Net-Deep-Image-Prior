import torch
import numpy as np

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

from tqdm import tqdm

from .loss import loss_BCD
from .networks.cnet import get_cnet
from .networks.mnet import get_mnet

from utils.callbacks import CallbacksList
from utils.utils_BCD import C_to_OD_torch


class DVBCDModel():
    def __init__(
                self, cnet_name="unet6", mnet_name="resnet_18_in", 
                sigmaRui_sq=torch.tensor([0.05, 0.05]), lambda_val=1.0, lr_cnet=1e-4, 
                lr_mnet=1e-4, lr_decay=0.1, clip_grad_cnet=np.Inf, clip_grad_mnet=np.Inf,
                device=torch.device('cpu')
                ):
        self.cnet = get_cnet(cnet_name)
        self.mnet = get_mnet(mnet_name, kernel_size=3)

        if torch.cuda.device_count() > 1:
            self.cnet = torch.nn.DataParallel(self.cnet)
            self.mnet = torch.nn.DataParallel(self.mnet)

        self.loss_fn = loss_BCD
        self.device = device

        #self.sigmaRui_sq = torch.tensor([args.sigmaRui_h_sq, args.sigmaRui_e_sq]).to(self.device)
        self.sigmaRui_sq = sigmaRui_sq
        self.lambda_val = lambda_val
        self.pretraining_lambda = 1.0

        self.lr_cnet = lr_cnet
        self.lr_mnet = lr_mnet
        self.lr_decay = lr_decay

        self.clip_grad_cnet = clip_grad_cnet
        self.clip_grad_mnet = clip_grad_mnet

        self.stop_training = False
        self.optim_initiated = False
        self.optimizers = None
        self.lr_schedulers = None

        self.callbacks_list = CallbacksList()
    
    def set_callbacks(self, callbacks):
        self.callbacks_list.set_callbacks(callbacks)

    def forward(self, y):
        out_Mnet_mean, out_Mnet_var = self.mnet(y) # shape: (batch_size, 3, 2), (batch_size, 1, 2)
        out_Cnet = self.cnet(y) # shape: (batch_size, 2, H, W)
        batch_size, c, heigth, width = out_Cnet.shape 
        patch_size = heigth # heigth = width = patch_size
        Cflat = out_Cnet.view(-1, 2, patch_size * patch_size) #shape: (batch, 2, patch_size * patch_size)
        Y_rec = torch.matmul(out_Mnet_mean, Cflat) #shape: (batch, 3, patch_size * patch_size)
        Y_rec = Y_rec.view(batch_size, 3, patch_size, patch_size) #shape: (batch, 3, patch_size, patch_size)
        return out_Mnet_mean, out_Mnet_var, out_Cnet, Y_rec

    def train(self):
        self.cnet.train()
        self.mnet.train()
    
    def eval(self):
        self.cnet.eval()
        self.mnet.eval()
    
    def to(self, device):
        self.cnet = self.cnet.to(device)
        self.mnet = self.mnet.to(device)

    def init_optimizers(self):
        #pre_optimizer_CNet = optim.Adam(self.cnet.parameters(), lr=5e-4)
        #pre_optimizer_MNet = optim.Adam(self.mnet.parameters(), lr=5e-4)
        Cnet_opt = torch.optim.Adam(self.cnet.parameters(), lr=self.lr_cnet)
        Mnet_opt = torch.optim.Adam(self.mnet.parameters(), lr=self.lr_mnet)
        
        Cnet_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(Cnet_opt, mode='min', factor=self.lr_decay, patience=5, verbose=True)
        Mnet_sch = torch.optim.lr_scheduler.ReduceLROnPlateau(Mnet_opt, mode='min', factor=self.lr_decay, patience=5, verbose=True)
        #Cnet_sch = torch.optim.lr_scheduler.StepLR(Cnet_opt, step_size=20, gamma=self.lr_decay)
        #Mnet_sch = torch.optim.lr_scheduler.StepLR(Mnet_opt, step_size=20, gamma=self.lr_decay)

        self.optimizers = (Cnet_opt, Mnet_opt)
        self.lr_schedulers = (Cnet_sch, Mnet_sch)
        self.optim_initiated = True

    def training_step(self, batch, batch_idx, pretraining=False):

        Cnet_opt, Mnet_opt = self.optimizers

        Y_RGB, Y_OD, MR = batch
        #Y_RGB = Y_RGB.to(self.device)
        Y_OD = Y_OD.to(self.device)
        MR = MR.to(self.device)
        self.sigmaRui_sq = self.sigmaRui_sq.to(self.device)

        Mnet_opt.zero_grad()
        Cnet_opt.zero_grad()

        out_Mnet_mean, out_Mnet_var, out_Cnet, Y_rec_od = self.forward(Y_OD) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)

        if pretraining:
            lambda_val = self.pretraining_lambda
        else:
            lambda_val = self.lambda_val
        loss, loss_kl, loss_mse = self.loss_fn(Y_OD, MR, Y_rec_od, out_Cnet, out_Mnet_mean, out_Mnet_var, self.sigmaRui_sq, lambda_val)

        loss.backward()
        
        #print(count_dict)

        #torch.nn.utils.clip_grad_norm_(self.cnet.parameters(), self.clip_grad_cnet)
        #torch.nn.utils.clip_grad_norm_(self.mnet.parameters(), self.clip_grad_mnet)

        Mnet_opt.step()
        Cnet_opt.step()

        #Y_rec_rgb = od2rgb_torch(undo_normalization(Y_rec_od))
        mse_rec = self.compute_mse(Y_OD, Y_rec_od)
        psnr_rec = self.compute_psnr(Y_OD, Y_rec_od)
        ssim_rec = self.compute_ssim(Y_OD, Y_rec_od)

        return {'train_loss' : loss.item(), 'train_loss_mse' : loss_mse.item(), 'train_loss_kl' : loss_kl.item(), 'train_mse_rec' : mse_rec.item(), 'train_psnr_rec' : psnr_rec.item(), 'train_ssim_rec' : ssim_rec.item()}

    def fit(self, max_epochs, train_dataloader, val_dataloader=None, pretraining=False):
        if val_dataloader is None:
            val_dataloader = train_dataloader

        self.sigmaRui_sq.to(self.device)
        self.to(self.device)
        
        if not self.optim_initiated:
            self.init_optimizers()
        
        epoch_log_dic = {}
        tmp_train_log_dic = {}
        tmp_val_log_dic = {}
        
        for epoch in range(1, max_epochs + 1):

            self.callbacks_list.on_epoch_begin(epoch)
            
            # Trainining loop:
            self.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            pbar.set_description(f"Epoch {epoch} - Training")
            for batch_idx, batch in pbar:
                self.callbacks_list.on_train_step_begin(self, batch_idx)
                m_dic = self.training_step(batch, batch_idx, pretraining=pretraining)
                self.callbacks_list.on_train_step_end(self)

                for k, v in m_dic.items():
                    if k not in tmp_train_log_dic.keys():
                        tmp_train_log_dic[k] = []
                    tmp_train_log_dic[k].append(v)

                if batch_idx == (len(train_dataloader) - 1):
                    pbar.set_postfix({k : str(round(np.mean(v),5)) for k, v in tmp_train_log_dic.items()})
                else:
                    pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()})            

            # Eval loop
            self.eval()
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            pbar.set_description(f"Epoch {epoch} - Validation")
            with torch.no_grad():
                for batch_idx, batch in pbar:
                    self.callbacks_list.on_val_step_begin(self, batch_idx)
                    m_dic = self.validation_step(batch, batch_idx)
                    self.callbacks_list.on_val_step_end(self)
                    pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()})
                    for k, v in m_dic.items():
                        if k not in tmp_val_log_dic.keys():
                            tmp_val_log_dic[k] = []
                        tmp_val_log_dic[k].append(v)

                    if batch_idx == (len(val_dataloader) - 1):
                        pbar.set_postfix({k : str(round(np.mean(v),5)) for k, v in tmp_val_log_dic.items()})
                    else:
                        pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()}) 

            for tmp_dict in [tmp_train_log_dic, tmp_val_log_dic]:
                for k, v in tmp_dict.items():
                    if k not in epoch_log_dic.keys():
                        epoch_log_dic[k] = []
                    epoch_log_dic[k] = np.mean(v)
                    tmp_dict[k] = []

            Cnet_sch, Mnet_sch = self.lr_schedulers
            if Cnet_sch is not None:
                Cnet_sch.step(epoch_log_dic['val_loss'])
            if Mnet_sch is not None:
                Mnet_sch.step(epoch_log_dic['val_loss'])

            self.callbacks_list.on_epoch_end(epoch, epoch_log_dic)
            if self.stop_training:
                break

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')
    
    def _shared_eval_step(self, batch, batch_idx, prefix):

        Y_RGB, Y_OD, MR = batch
        #Y_RGB = Y_RGB.to(self.device)
        Y_OD = Y_OD.to(self.device)
        MR = MR.to(self.device)

        out_Mnet_mean, out_Mnet_var, out_Cnet, Y_rec_od = self.forward(Y_OD) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)

        loss, loss_kl, loss_mse = self.loss_fn(Y_OD, MR, Y_rec_od, out_Cnet, out_Mnet_mean, out_Mnet_var, self.sigmaRui_sq, self.lambda_val)

        #Y_rec_rgb = od2rgb_torch(undo_normalization(Y_rec_od))
        mse_rec = self.compute_mse(Y_OD, Y_rec_od)
        psnr_rec = self.compute_psnr(Y_OD, Y_rec_od)
        ssim_rec = self.compute_ssim(Y_OD, Y_rec_od)

        return {
            f'{prefix}_loss' : loss.item(), f'{prefix}_loss_mse' : loss_mse.item(), f'{prefix}_loss_kl' : loss_kl.item(), 
            f'{prefix}_mse_rec' : mse_rec.item(), f'{prefix}_psnr_rec' : psnr_rec.item(), f'{prefix}_ssim_rec' : ssim_rec.item()
            }
    
    def validation_step_GT(self, batch, batch_idx):
        return self._shared_eval_step_GT(batch, batch_idx, 'val')
    
    def test_step_GT(self, batch, batch_idx):
        return self._shared_eval_step_GT(batch, batch_idx, 'test')
    
    def _shared_eval_step_GT(self, batch, batch_idx, prefix):

        Y_RGB, Y_od, MR, C_GT, M_GT = batch
        Y_od = Y_od.to(self.device)
        MR = MR.to(self.device)
        C_GT = C_GT.to(self.device)
        M_GT = M_GT.to(self.device)

        out_Mnet_mean, out_Mnet_var, out_Cnet, Y_rec = self.forward(Y_od) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)

        loss, loss_kl, loss_mse = self.loss_fn(Y_od, MR, Y_rec, out_Cnet, out_Mnet_mean, out_Mnet_var, self.sigmaRui_sq, self.lambda_val)

        psnr_rec = self.compute_psnr(Y_od, Y_rec)
        ssim_rec = self.compute_ssim(Y_od, Y_rec)

        C_OD = C_to_OD_torch(out_Cnet, out_Mnet_mean)
        H_OD = C_OD[:, 0, :, :]
        E_OD = C_OD[:, 1, :, :]

        C_GT_OD = C_to_OD_torch(C_GT, M_GT)
        H_OD_GT =  C_GT_OD[:, 0, :, :]
        E_OD_GT =  C_GT_OD[:, 1, :, :]

        psnr_gt_h = self.compute_psnr(H_OD, H_OD_GT)
        psnr_gt_e = self.compute_psnr(E_OD, E_OD_GT)
        ssim_gt_h = self.compute_ssim(H_OD, H_OD_GT)
        ssim_gt_e = self.compute_ssim(E_OD, E_OD_GT)

        return {
            f'{prefix}_loss' : loss.item(), f'{prefix}_loss_mse' : loss_mse.item(), f'{prefix}_loss_kl' : loss_kl.item(), 
            f'{prefix}_psnr_rec' : psnr_rec.item(), f'{prefix}_ssim' : ssim_rec.item(), 
            f'{prefix}_psnr_gt_h' : psnr_gt_h.item(), f'{prefix}_psnr_gt_e' : psnr_gt_e.item(),
            f'{prefix}_ssim_gt_h' : ssim_gt_h.item(), f'{prefix}_ssim_gt_e' : ssim_gt_e.item()
            }

    def evaluate(self, test_dataloader):
        self.eval()
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        pbar.set_description(f"Testing")
        tmp_metrics_dic = {}
        with torch.no_grad():
            for batch_idx, batch in pbar:
                m_dic = self.test_step(batch, batch_idx)
                pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()})
                for k, v in m_dic.items():
                    if k not in tmp_metrics_dic.keys():
                        tmp_metrics_dic[k] = []
                    tmp_metrics_dic[k].append(v)
                if batch_idx == (len(test_dataloader) - 1):
                    pbar.set_postfix({k : str(round(np.mean(v),5)) for k, v in tmp_metrics_dic.items()})
                else:
                    pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()}) 
        return {k : np.mean(v) for k, v in tmp_metrics_dic.items()}
    
    def evaluate_GT(self, test_dataloader):
        self.eval()
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        pbar.set_description(f"Testing")
        tmp_metrics_dic = {}
        with torch.no_grad():
            for batch_idx, batch in pbar:
                m_dic = self.test_step_GT(batch, batch_idx)
                pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()})
                for k, v in m_dic.items():
                    if k not in tmp_metrics_dic.keys():
                        tmp_metrics_dic[k] = []
                    tmp_metrics_dic[k].append(v)

                if batch_idx == (len(test_dataloader) - 1):
                    pbar.set_postfix({k : str(round(np.mean(v),5)) for k, v in tmp_metrics_dic.items()})
                else:
                    pbar.set_postfix({k : str(round(v, 4)) for k, v in m_dic.items()}) 
        return {k : np.mean(v) for k, v in tmp_metrics_dic.items()}
    
    def compute_psnr(self, Y, Y_rec):
        return peak_signal_noise_ratio(Y_rec, Y)
    
    def compute_ssim(self, Y, Y_rec):
        return structural_similarity_index_measure(Y_rec, Y)
    
    def compute_mse(self, Y, Y_rec):
        return torch.nn.functional.mse_loss(Y_rec, Y, reduction='mean')
    
    def save(self, path):
        rest = path.split("/")[0:-1]
        name = path.split("/")[-1]
        final_path_cnet = "/".join(rest) + f"/cnet_{name}"
        final_path_mnet = "/".join(rest) + f"/mnet_{name}"
        torch.save(self.cnet.state_dict(), final_path_cnet)
        torch.save(self.mnet.state_dict(), final_path_mnet)
    
    def load(self, path):
        rest = path.split("/")[0:-1]
        name = path.split("/")[-1]
        final_path_cnet = "/".join(rest) + f"/cnet_{name}"
        final_path_mnet = "/".join(rest) + f"/mnet_{name}"
        self.cnet.load_state_dict(torch.load(final_path_cnet))
        self.mnet.load_state_dict(torch.load(final_path_mnet))