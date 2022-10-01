import torch
import numpy as np

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from tqdm import tqdm

from .loss import loss_BCD
from .networks.cnet import get_cnet
from .networks.mnet import get_mnet

from utils.callbacks import CallbacksList


class DVBCDModel():
    def __init__(
                self, cnet_name="unet6", mnet_name="resnet_18_in", 
                sigmaRui_sq=torch.tensor([0.05, 0.05]), theta=0.5, lr_cnet=1e-4, 
                lr_mnet=1e-4, lr_decay=0.1, clip_grad_cnet=1e5, clip_grad_mnet=1e5,
                device=torch.device('cpu')
                ):
        self.cnet = get_cnet(cnet_name)
        self.mnet = get_mnet(mnet_name, kernel_size=3)
        self.loss_fn = loss_BCD
        self.device = device

        #self.sigmaRui_sq = torch.tensor([args.sigmaRui_h_sq, args.sigmaRui_e_sq]).to(self.device)
        self.sigmaRui_sq = sigmaRui_sq.to(self.device)
        self.theta = theta
        self.pretraining_theta = 1-1e-6

        self.lr_cnet = lr_cnet
        self.lr_mnet = lr_mnet
        self.lr_decay = lr_decay

        self.clip_grad_cnet = clip_grad_cnet
        self.clip_grad_mnet = clip_grad_mnet

        self.stop_training = False
        self.optim_initiated = False
        self.optimizers = None
        self.lr_schedulers = None

        self.psnr_obj = PeakSignalNoiseRatio().to(self.device)
        self.ssim_obj = StructuralSimilarityIndexMeasure().to(self.device)

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
        self.cnet.to(device)
        self.mnet.to(device)

    def init_optimizers(self):
        #pre_optimizer_CNet = optim.Adam(self.cnet.parameters(), lr=5e-4)
        #pre_optimizer_MNet = optim.Adam(self.mnet.parameters(), lr=5e-4)
        Cnet_opt = torch.optim.Adam(self.cnet.parameters(), lr=self.lr_cnet)
        Mnet_opt = torch.optim.Adam(self.mnet.parameters(), lr=self.lr_mnet)
        Cnet_sch = torch.optim.lr_scheduler.StepLR(Cnet_opt, step_size=20, gamma=self.lr_decay)
        Mnet_sch = torch.optim.lr_scheduler.StepLR(Mnet_opt, step_size=20, gamma=self.lr_decay)

        self.optimizers = (Cnet_opt, Mnet_opt)
        self.lr_schedulers = (Cnet_sch, Mnet_sch)
        self.optim_initiated = True

    def training_step(self, batch, batch_idx, pretraining=False):

        Cnet_opt, Mnet_opt = self.optimizers

        _, Y, MR = batch
        Y = Y.to(self.device)
        MR = MR.to(self.device)

        Cnet_opt.zero_grad()
        Mnet_opt.zero_grad()

        out_Mnet_mean, out_Mnet_var, out_Cnet, Y_rec = self.forward(Y) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)
        psnr = self.compute_psnr(Y, Y_rec)
        ssim = self.compute_ssim(Y, Y_rec)

        self.sigmaRui_sq.to(self.device)
        if pretraining:
            theta = self.pretraining_theta
        else:
            theta = self.theta
        loss, loss_kl, loss_mse = self.loss_fn(Y, MR, Y_rec, out_Cnet, out_Mnet_mean, out_Mnet_var, self.sigmaRui_sq, theta)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.cnet.parameters(), self.clip_grad_cnet)
        torch.nn.utils.clip_grad_norm_(self.mnet.parameters(), self.clip_grad_mnet)

        Cnet_opt.step()
        Mnet_opt.step()

        Cnet_sch, Mnet_sch = self.lr_schedulers
        Cnet_sch.step()
        Mnet_sch.step()

        return {'train_loss' : loss.item(), 'train_loss_mse' : loss_mse.item(), 'train_loss_kl' : loss_kl.item(), 'train_psnr' : psnr.item(), 'train_ssim' : ssim.item()}
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')
    
    def _shared_eval_step(self, batch, batch_idx, prefix):

        _, Y, MR = batch
        Y = Y.to(self.device)
        MR = MR.to(self.device)

        out_Mnet_mean, out_Mnet_var, out_Cnet, Y_rec = self.forward(Y) # shape: (batch_size, 3, 2), (batch_size, 1, 2), (batch_size, 2, H, W)

        loss, loss_kl, loss_mse = self.loss_fn(Y, MR, Y_rec, out_Cnet, out_Mnet_mean, out_Mnet_var, self.sigmaRui_sq, self.theta)

        psnr = self.compute_psnr(Y, Y_rec)
        ssim = self.compute_ssim(Y, Y_rec)

        return {f'{prefix}_loss' : loss.item(), f'{prefix}_loss_mse' : loss_mse.item(), f'{prefix}_loss_kl' : loss_kl.item(), f'{prefix}_psnr' : psnr.item(), f'{prefix}_ssim' : ssim.item()}
    
    def fit(self, max_epochs, train_dataloader, val_dataloader=None, pretraining=False):
        if val_dataloader is None:
            val_dataloader = train_dataloader
        
        if not self.optim_initiated:
            self.init_optimizers()
        
        epoch_log_dic = {
            'train_loss': [], 'train_loss_mse' : [], 'train_loss_kl' : [], 'train_psnr' : [], 'train_ssim' : [],
            'val_loss': [], 'val_loss_mse' : [], 'val_loss_kl' : [], 'val_psnr' : [], 'val_ssim' : []
        }
        tmp_log_dic = {k : [] for k in epoch_log_dic.keys()}

        self.sigmaRui_sq.to(self.device)
        self.to(self.device)
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
                pbar.set_postfix(m_dic)
                for k, v in m_dic.items():
                    tmp_log_dic[k].append(v)

            # Eval loop
            self.eval()
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            pbar.set_description(f"Epoch {epoch} - Validation")
            with torch.no_grad():
                for batch_idx, batch in pbar:
                    self.callbacks_list.on_val_step_begin(self, batch_idx)
                    m_dic = self.validation_step(batch, batch_idx)
                    self.callbacks_list.on_val_step_end(self)
                    pbar.set_postfix(m_dic)
                    for k, v in m_dic.items():
                        tmp_log_dic[k].append(v)
            
            for k, v in tmp_log_dic.items():
                epoch_log_dic[k] = np.mean(v)
                tmp_log_dic[k] = []

            self.callbacks_list.on_epoch_end(epoch, epoch_log_dic)
            if self.stop_training:
                break

    def evaluate(self, test_dataloader):
        self.eval()
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        pbar.set_description(f"Testing")
        with torch.no_grad():
            for batch_idx, batch in pbar:
                m_dic = self.test_step(batch, batch_idx)
                pbar.set_postfix(m_dic)
        return m_dic
    
    def compute_psnr(self, Y, Y_rec):
        return self.psnr_obj(Y_rec, Y)
    
    def compute_ssim(self, Y, Y_rec):
        return self.ssim_obj(Y_rec, Y)