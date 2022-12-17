import os
import torch

from utils.callbacks import EarlyStopping, ModelCheckpoint, History
from options import set_train_opts

from utils.utils_data import get_train_dataloaders
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

SAVE_MODEL_NAME = f"{args.mnet_name}_{args.pretraining_epochs}pe_{args.patch_size}ps_{args.lambda_val}lambda_{args.sigmaRui_sq}sigmaRui_{args.n_samples}nsamples"
SAVE_MODEL_PATH = os.path.join(args.save_model_dir, f"{SAVE_MODEL_NAME}/")
HISTORY_PATH = os.path.join(args.save_history_dir, f"{SAVE_MODEL_NAME}.csv")

if not os.path.exists(args.save_history_dir):
    os.makedirs(args.save_history_dir)

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)


train_dataloader, val_dataloader = get_train_dataloaders(args.camelyon_data_path, args.patch_size, args.batch_size, args.num_workers, val_prop=args.val_prop, n_samples=args.n_samples)

sigmaRui_sq = torch.tensor([args.sigmaRui_sq, args.sigmaRui_sq])
model = DVBCDModel(
                cnet_name=args.cnet_name, mnet_name=args.mnet_name, 
                sigmaRui_sq=sigmaRui_sq, lambda_val=args.lambda_val, lr_cnet=args.lr_cnet, lr_mnet=args.lr_mnet,
                lr_decay=args.lr_decay, clip_grad_cnet=args.clip_grad_cnet, clip_grad_mnet=args.clip_grad_mnet,
                device=DEVICE
                )
callbacks = [
    EarlyStopping(model, score_name="val_loss", mode="min", delta=0.0, patience=args.patience, path=SAVE_MODEL_PATH), 
    ModelCheckpoint(model, path=SAVE_MODEL_PATH, save_freq=args.save_freq), 
    History(path = HISTORY_PATH)]
model.set_callbacks(callbacks)
if args.pretraining_epochs > 0:
    model.fit(args.pretraining_epochs, train_dataloader, val_dataloader, pretraining=True)
    model.init_optimizers()
model.fit(args.epochs, train_dataloader, val_dataloader, pretraining=False)

