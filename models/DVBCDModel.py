import torch

#from torchmetrics import PeakSignalNoiseRatio

from tqdm import tqdm

from .loss import loss_BCD
from .networks.cnet import get_cnet
from .networks.mnet import get_mnet


class DVBCDModel():
    def __init__(self, args, device=torch.device('cpu')):
        self.args = args
        self.cnet = get_cnet(args.CNet)
        self.mnet = get_mnet(args.MNet, kernel_size=3)
        self.loss_fn = loss_BCD
        self.device = device

        self.sigmaRui_sq = torch.tensor([args.sigmaRui_h_sq, args.sigmaRui_e_sq]).to(self.device)
        self.theta = 0.5
        self.pre_mse = args.pre_mse
        self.pre_kl = args.pre_kl

        self.optim_initiated = False
        self.optimizers = None
        self.lr_schedulers = None
        
    def forward(self, y):
        out_MNet_mean, out_Mnet_var = self.mnet(y) # shape: (batch_size, 3, 2), (batch_size, 1, 2)
        out_CNet = self.cnet(y) # shape: (batch_size, 2, H, W)
        return out_MNet_mean, out_Mnet_var, out_CNet

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
        Cnet_opt = torch.optim.Adam(self.cnet.parameters(), lr=5e-4)
        Mnet_opt = torch.optim.Adam(self.mnet.parameters(), lr=5e-4)
        Cnet_sch = torch.optim.lr_scheduler.StepLR(Cnet_opt, step_size=20, gamma=0.1)
        Mnet_sch = torch.optim.lr_scheduler.StepLR(Mnet_opt, step_size=20, gamma=0.1)

        self.optimizers = (Cnet_opt, Mnet_opt)
        self.lr_schedulers = (Cnet_sch, Mnet_sch)
        self.optim_initiated = True

    def training_step(self, batch, batch_idx, pretraining=False):

        Cnet_opt, Mnet_opt = self.optimizers

        y, mR = batch
        y = y.to(self.device)
        mR = mR.to(self.device)

        Cnet_opt.zero_grad()
        Mnet_opt.zero_grad()

        out_Mnet_mean, out_Mnet_var = self.mnet(y) # shape: (batch_size, 3, 2), (batch_size, 1, 2)
        out_Cnet = self.cnet(y) # shape: (batch_size, 2, H, W)

        #loss, loss_mse, loss_kl, loss_kl_h, loss_kl_e = self.loss_fn(  out_CNet, out_MNet_mean, out_Mnet_var, y, mR,
        #                                    self.args.sigmaRui_h_sq, self.args.sigmaRui_e_sq, 
        #                                    pretraining=True, pre_mse = self.args.pre_mse, pre_kl = self.args.pre_kl
        #                                    )
        
        self.sigmaRui_sq.to(self.device)
        loss, loss_kl, loss_mse = self.loss_fn(out_Cnet, out_Mnet_mean, out_Mnet_var, y, self.sigmaRui_sq, mR, self.theta, pretraining=pretraining, pre_mse = self.pre_mse, pre_kl = self.pre_kl)

        loss.backward()
        Cnet_opt.step()
        Mnet_opt.step()

        CNet_sch, MNet_sch = self.lr_schedulers
        CNet_sch.step()
        MNet_sch.step()

        return {'train_loss' : loss.item(), 'train_loss_mse' : loss_mse.item(), 'train_loss_kl' : loss_kl.item()}
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, 'test')
    
    def _shared_eval_step(self, batch, batch_idx, prefix):

        y, mR = batch
        y = y.to(self.device)
        mR = mR.to(self.device)

        out_Mnet_mean, out_Mnet_var = self.mnet(y) # shape: (batch_size, 3, 2), (batch_size, 1, 2)
        out_Cnet = self.cnet(y) # shape: (batch_size, 2, H, W)

        #loss, loss_mse, loss_kl, loss_kl_h, loss_kl_e = self.loss_fn(  out_CNet, out_MNet_mean, out_Mnet_var, y, mR,
        #                                    self.args.sigmaRui_h_sq, self.args.sigmaRui_e_sq, 
        #                                    pretraining=True, pre_mse = self.args.pre_mse, pre_kl = self.args.pre_kl
        #                                    )
        
        loss, loss_kl, loss_mse = self.loss_fn(out_Cnet, out_Mnet_mean, out_Mnet_var, y, self.sigmaRui_sq, mR, self.theta)

        #psnr = PeakSignalNoiseRatio()
        return {f'{prefix}_loss' : loss.item(), f'{prefix}_loss_mse' : loss_mse.item(), f'{prefix}_loss_kl' : loss_kl.item()}
    
    def fit(self, max_epochs, train_dataloader, val_dataloader=None, pretraining=False):
        if val_dataloader is None:
            val_dataloader = train_dataloader
        
        if not self.optim_initiated:
            self.init_optimizers()
        
        self.sigmaRui_sq.to(self.device)
        self.to(self.device)
        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}")
            
            # Trainining loop:
            self.train()
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            pbar.set_description(f"Epoch {epoch} - Training")
            for batch_idx, batch in pbar:
                m_dic = self.training_step(batch, batch_idx, pretraining=pretraining)
                pbar.set_postfix(m_dic)

            # Eval loop
            self.eval()
            pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            pbar.set_description(f"Epoch {epoch} - Validation")
            for batch_idx, batch in pbar:
                m_dic = self.validation_step(batch, batch_idx)
                pbar.set_postfix(m_dic)

    def evaluate(self, test_dataloader):
        self.eval()
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        pbar.set_description(f"Testing")
        for batch_idx, batch in pbar:
            m_dic = self.test_step(batch, batch_idx)
            pbar.set_postfix(m_dic)
        return m_dic