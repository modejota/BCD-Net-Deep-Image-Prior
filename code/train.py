import os
import torch

from utils.callbacks import EarlyStopping, ModelCheckpoint, History
from options import set_train_opts

from utils.utils_data import get_train_dataloaders, get_wssb_dataloader
from models.DVBCDModel import DVBCDModel

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

USE_GPU = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('using device:', DEVICE)

args = set_train_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

MAIN_PATH = "/work/work_fran/Deep_Var_BCD/"

SAVE_MODEL_NAME = f"{args.mnet_name}_{args.pretraining_epochs}pe_{args.patch_size}ps_{args.theta_val}theta_{args.sigmaRui_sq}sigmaRui_{args.n_samples}nsamples"
SAVE_MODEL_PATH = os.path.join(args.save_model_dir, f"{SAVE_MODEL_NAME}/")
HISTORY_PATH = os.path.join(args.save_history_dir, f"{SAVE_MODEL_NAME}.csv")
VAL_TYPE = args.val_type
ES_METRIC="val_loss"

if not os.path.exists(args.save_history_dir):
    os.makedirs(args.save_history_dir)

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

train_dataloader, val_dataloader = get_train_dataloaders(args.camelyon_data_path, args.patch_size, args.batch_size, args.num_workers, val_prop=args.val_prop, n_samples=args.n_samples)
if VAL_TYPE == "GT":
    print("Using WSSB dataset for validation")
    ES_METRIC = "val_mse_gt"
    val_dataloader = get_wssb_dataloader(args.wssb_data_path, args.num_workers)

sigmaRui_sq = torch.tensor([args.sigmaRui_sq, args.sigmaRui_sq])
model = DVBCDModel(
                cnet_name=args.cnet_name, mnet_name=args.mnet_name, 
                sigmaRui_sq=sigmaRui_sq, theta_val=args.theta_val, lr_cnet=args.lr_cnet, lr_mnet=args.lr_mnet,
                lr_decay=args.lr_decay, clip_grad_cnet=args.clip_grad_cnet, clip_grad_mnet=args.clip_grad_mnet,
                device=DEVICE
                )

cnet_n_params = sum(p.numel() for p in model.cnet.parameters() if p.requires_grad)
mnet_n_params = sum(p.numel() for p in model.mnet.parameters() if p.requires_grad)
print(f"Number of trainable parameters in Cnet: {cnet_n_params}")
print(f"Number of trainable parameters in Mnet: {mnet_n_params}")

callbacks = [
    EarlyStopping(model, score_name=ES_METRIC, mode="min", delta=0.0, patience=args.patience, path=SAVE_MODEL_PATH), 
    ModelCheckpoint(model, path=SAVE_MODEL_PATH, save_freq=args.save_freq), 
    History(path = HISTORY_PATH)]
model.set_callbacks(callbacks)
if args.pretraining_epochs > 0:
    model.fit(args.pretraining_epochs, train_dataloader, val_dataloader, pretraining=True, val_type=VAL_TYPE)
    model.init_optimizers()
model.fit(args.epochs, train_dataloader, val_dataloader, pretraining=False, val_type=VAL_TYPE)

