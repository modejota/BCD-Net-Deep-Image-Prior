import os
import torch

from utils.callbacks import EarlyStopping, History
from options import set_train_opts

from utils.utils_data import get_train_dataloaders, get_test_dataloaders
from models.DVBCDModel import DVBCDModel

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

USE_GPU = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda:3')
else:
    DEVICE = torch.device('cpu')
print('using device:', DEVICE)

args = set_train_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))


train_dataloader, val_dataloader = get_train_dataloaders(args.camelyon_data_path, args.patch_size, args.batch_size, args.num_workers, val_prop=args.val_prop, n_samples=args.n_samples)

sigmaRui_sq = torch.tensor([args.sigmaRui_h_sq, args.sigmaRui_e_sq])
model = DVBCDModel(
                cnet_name="unet_6", mnet_name="resnet_18_in", 
                sigmaRui_sq=sigmaRui_sq, theta=args.theta, lr_cnet=args.lr_cnet, 
                lr_mnet=args.lr_mnet, lr_decay=args.lr_decay, clip_grad_cnet=args.clip_grad_cnet, clip_grad_mnet=args.clip_grad_mnet,
                device=DEVICE
                )
callbacks = [EarlyStopping(model, score_name="val_loss", patience=10, path=args.save_model_dir)]
model.set_callbacks(callbacks)
model.fit(args.pretraining_epochs, train_dataloader, val_dataloader, pretraining=True)
model.init_optimizers()
model.fit(args.epochs, train_dataloader, val_dataloader, pretraining=False)

test_dataloader_camelyon, test_dataloader_wssb_dict = get_test_dataloaders(args.wssb_data_path, args.patch_size, args.batch_size, args.num_workers, val_prop=args.val_prop, n_samples=None)
# Falta calcular las metricas de test para WSSB y Camelyon


