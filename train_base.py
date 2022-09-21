import time
import os
import sys
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# import scipy.io
# from utils.utils_metric import batch_PSNR, batch_SSIM
from datasets.main_dataset import get_dataloaders
from loss import loss_BCD
from networks.cnet import get_cnet
from networks.mnet import get_mnet
from options import set_opts

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

USE_GPU = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:2')
else:
    device = torch.device('cpu')
print('using device:', device)

args = set_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

def adjust_learning_rate(optimizer, epoch, args):
    if epoch <= 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    elif epoch <= 80:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-6

def main():
    sigma_s2 = torch.tensor([args.sigma_h2, args.sigma_e2])
    sigma_s2 = sigma_s2.to(device)

    cnet = get_cnet(args.CNet)
    mnet = get_mnet(args.MNet, kernel_size=3)
    cnet = cnet.to(device)
    mnet = mnet.to(device)
    optimizer_c = optim.Adam(cnet.parameters(), args.lr_C)
    optimizer_m = optim.Adam(mnet.parameters(), args.lr_M)

    data_loaders = get_dataloaders(args)
    pre_optimizer_m = optim.Adam(mnet.parameters(), lr=5e-4)
    args.epoch_start = 0



    for epoch in range(args.epoch_start, args.epochs):
        tic = time.time()
        cnet.train()
        mnet.train()

        adjust_learning_rate(optimizer_m, epoch, args)
        adjust_learning_rate(optimizer_c, epoch, args)

        lr_C = optimizer_c.param_groups[0]['lr']
        lr_M = optimizer_m.param_groups[0]['lr']

        for ii, data in enumerate(data_loaders['Train']):
            Y = data[0].to(device)
            MR = data[1].to(device)

            optimizer_m.zero_grad()
            optimizer_c.zero_grad()

            out_Mnet_mean, out_Mnet_var = mnet(Y)
            out_Cnet = cnet(Y)

            loss, loss_kl, loss_mse = loss_BCD(out_Cnet, out_Mnet_mean, out_Mnet_var, Y, sigma_s2, MR, args.patch_size)

            loss.backward()
            optimizer_m.step()
            optimizer_c.step()