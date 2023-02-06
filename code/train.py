import os
import torch

from utils.callbacks import EarlyStopping, ModelCheckpoint, History
from options import set_train_opts

from utils.datasets import CamelyonDataset, WSSBDatasetTest
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

SAVE_MODEL_NAME = f"{args.cnet_name}_{args.mnet_name}_{args.pretraining_epochs}pe_{args.patch_size}ps_{args.theta_val}theta_{args.sigmaRui_sq}sigmaRui_{args.n_samples}nsamples"
SAVE_MODEL_PATH = os.path.join(args.save_model_dir, f"{SAVE_MODEL_NAME}/")
HISTORY_PATH = os.path.join(args.save_history_dir, f"{SAVE_MODEL_NAME}.csv")
ES_METRIC="val_loss"
TRAIN_CENTERS =  [0,2,4]
VAL_CENTERS = [1,3]

if not os.path.exists(args.save_history_dir):
    os.makedirs(args.save_history_dir)

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

cam_train_dataset = CamelyonDataset(args.camelyon_data_path, TRAIN_CENTERS, patch_size=args.patch_size, n_samples=args.n_samples)
if args.val_type == "GT":
    print("Using WSSB dataset for validation")
    ES_METRIC = "val_mse_gt"
    train_dataset = cam_train_dataset
    val_dataset = WSSBDatasetTest(args.wssb_data_path, organ_list=['Lung', 'Breast', 'Colon'])
    val_batch_size = 1
else:
    print("Using Camelyon dataset for validation")
    ES_METRIC = "val_mse_rec"
    train_dataset = cam_train_dataset
    n_samples_val = int(args.val_prop * len(cam_train_dataset))
    val_dataset = CamelyonDataset(args.camelyon_data_path, VAL_CENTERS, patch_size=args.patch_size, n_samples=args.n_samples)
    val_batch_size = args.batch_size

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=args.num_workers)
    
sigmaRui_sq = torch.tensor([args.sigmaRui_sq, args.sigmaRui_sq])
model = DVBCDModel(
                cnet_name=args.cnet_name, mnet_name=args.mnet_name, 
                sigmaRui_sq=sigmaRui_sq, theta_val=args.theta_val, 
                lr=args.lr, lr_decay=args.lr_decay, clip_grad=args.clip_grad
                )

cnet_n_params = sum(p.numel() for p in model.module.cnet.parameters() if p.requires_grad)
mnet_n_params = sum(p.numel() for p in model.module.mnet.parameters() if p.requires_grad)
print(f"Number of trainable parameters in Cnet: {cnet_n_params}")
print(f"Number of trainable parameters in Mnet: {mnet_n_params}")

callbacks = [
    EarlyStopping(model, score_name=ES_METRIC, mode="min", delta=0.0, patience=args.patience, path=SAVE_MODEL_PATH), 
    ModelCheckpoint(model, path=SAVE_MODEL_PATH, save_freq=args.save_freq), 
    History(path = HISTORY_PATH)]
model.set_callbacks(callbacks)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model.DP()
model.to(DEVICE)

if args.pretraining_epochs > 0:
    model.fit(args.pretraining_epochs, train_dataloader, val_dataloader, pretraining=True)
    model.init_optimizer()
model.fit(args.epochs, train_dataloader, val_dataloader, pretraining=False)

