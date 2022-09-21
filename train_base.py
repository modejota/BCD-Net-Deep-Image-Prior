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
from loss import loss_fn
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

def main():
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