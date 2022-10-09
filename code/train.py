import os
import torch
import pandas as pd

from utils.callbacks import EarlyStopping, ModelCheckpoint
from options import set_train_opts

from utils.utils_data import get_train_dataloaders, get_camelyon_test_dataloader, get_wssb_test_dataloader
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

SAVE_KEYS = ["patch_size", "pretraining_epochs", "n_samples", "lambda_val"]

MAIN_PATH = "/work/work_fran/Deep_Var_BCD/"
RESULTS_PATH_CAMELYON = os.path.join(MAIN_PATH, args.results_dir, f"results_camelyon.csv")
RESULTS_PATH_WSSB = os.path.join(MAIN_PATH, args.results_dir, f"results_wssb.csv")

MODEL_DIR_NAME = f"{args.pretraining_epochs}pe_{args.patch_size}ps_{args.lambda_val}lambda_{args.n_samples}nsamples"
SAVE_MODEL_PATH = os.path.join(args.save_model_dir, f"{MODEL_DIR_NAME}/")

if not os.path.exists(SAVE_MODEL_PATH):
    os.makedirs(SAVE_MODEL_PATH)

train_dataloader, val_dataloader = get_train_dataloaders(args.camelyon_data_path, args.patch_size, args.batch_size, args.num_workers, val_prop=args.val_prop, n_samples=args.n_samples)

sigmaRui_sq = torch.tensor([args.sigmaRui_h_sq, args.sigmaRui_e_sq])
model = DVBCDModel(
                cnet_name="unet_6", mnet_name="resnet_18_in", 
                sigmaRui_sq=sigmaRui_sq, lambda_val=args.lambda_val, lr_cnet=args.lr_cnet, lr_mnet=args.lr_mnet,
                lr_decay=args.lr_decay, clip_grad_cnet=args.clip_grad_cnet, clip_grad_mnet=args.clip_grad_mnet,
                device=DEVICE
                )
callbacks = [EarlyStopping(model, score_name="val_loss", patience=args.patience, path=SAVE_MODEL_PATH), ModelCheckpoint(model, path=SAVE_MODEL_PATH, save_freq=args.save_freq)]
model.set_callbacks(callbacks)
if args.pretraining_epochs > 0:
    model.fit(args.pretraining_epochs, train_dataloader, val_dataloader, pretraining=True)
    model.init_optimizers()
model.fit(args.epochs, train_dataloader, val_dataloader, pretraining=False)

test_dataloader_camelyon = get_camelyon_test_dataloader(args.camelyon_data_path, args.patch_size, args.num_workers, args.n_samples)
test_dataloader_wssb_dict = get_wssb_test_dataloader(args.wssb_data_path, args.num_workers)

results_dic_camelyon = {}
results_dic_wssb = {}
for k in SAVE_KEYS:
    results_dic_camelyon[k] = getattr(args, k)
    results_dic_wssb[k] = getattr(args, k)

metrics_camelyon = model.evaluate(test_dataloader_camelyon)
for k in metrics_camelyon.keys():
    results_dic_camelyon["camelyon_" + k] = metrics_camelyon[k]

for organ in test_dataloader_wssb_dict.keys():
    metrics_wssb = model.evaluate_GT(test_dataloader_wssb_dict[organ])
    for k in metrics_wssb.keys():
        results_dic_wssb[f"wssb_{organ}_" + k] = metrics_wssb[k]


res_camelyon_df = pd.DataFrame(results_dic_camelyon, index=[0])
if(os.path.isfile(RESULTS_PATH_CAMELYON)):
    res_camelyon_df.to_csv(RESULTS_PATH_CAMELYON, mode='a', index=False, header=False)
else:
   res_camelyon_df.to_csv(RESULTS_PATH_CAMELYON, index=False)

res_wssb_df = pd.DataFrame(results_dic_wssb, index=[0])
if(os.path.isfile(RESULTS_PATH_WSSB)):
    res_wssb_df.to_csv(RESULTS_PATH_WSSB, mode='a', index=False, header=False)
else:
   res_wssb_df.to_csv(RESULTS_PATH_WSSB, index=False)


