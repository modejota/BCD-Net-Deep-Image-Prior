import os
import torch
import pandas as pd

from options import set_test_opts

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

args = set_test_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

SAVE_KEYS = ["mnet_name", "patch_size", "pretraining_epochs", "n_samples", "theta_val", "sigmaRui_sq"]

MAIN_PATH = "/work/work_fran/Deep_Var_BCD/"
RESULTS_PATH_CAMELYON = os.path.join(MAIN_PATH, args.results_dir, f"results_camelyon.csv")
RESULTS_PATH_WSSB = os.path.join(MAIN_PATH, args.results_dir, f"results_wssb.csv")

MODEL_DIR_NAME = f"{args.mnet_name}_{args.pretraining_epochs}pe_{args.patch_size}ps_{args.theta_val}theta_{args.sigmaRui_sq}sigmaRui_{args.n_samples}nsamples"
LOAD_MODEL_PATH = os.path.join(args.load_model_dir, f"{MODEL_DIR_NAME}/")

TEST_CENTERS = [1,3]

model = DVBCDModel(cnet_name=args.cnet_name, mnet_name=args.mnet_name, theta_val = args.theta_val)
model.load(LOAD_MODEL_PATH + "best.pt")

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model.DP()
model.to(DEVICE)

cam_dataset = CamelyonDataset(args.camelyon_data_path, TEST_CENTERS, patch_size=args.patch_size, n_samples=0.2*args.n_samples)
cam_dataloader = torch.utils.data.DataLoader(cam_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

wssb_dataloader_dic = {}
for organ in ['Lung', 'Breast', 'Colon']:
    wssb_dataset = WSSBDatasetTest(args.wssb_data_path, organ_list=[organ])
    wssb_dataloader_dic[organ] = torch.utils.data.DataLoader(args.wssb_data_path, batch_size=1, shuffle=False, num_workers=args.num_workers)

results_dic_camelyon = {}
results_dic_wssb = {}
for k in SAVE_KEYS:
    results_dic_camelyon[k] = getattr(args, k)
    results_dic_wssb[k] = getattr(args, k)

metrics_camelyon = model.evaluate(cam_dataloader)
for k in metrics_camelyon.keys():
    results_dic_camelyon["camelyon_" + k] = metrics_camelyon[k]

for organ in wssb_dataloader_dic.keys():
    metrics_wssb = model.evaluate(wssb_dataloader_dic[organ])
    for k in metrics_wssb.keys():
        results_dic_wssb[f"wssb_{organ}_" + k] = metrics_wssb[k]

print("Writing results to csv...")

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