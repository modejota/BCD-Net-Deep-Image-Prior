import os
import torch
import h5py
from tqdm import tqdm

from utils.utils_data import get_files_dataloader
from models.DVBCDModel import DVBCDModel
from options import set_deconvolve_opts

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

args = set_deconvolve_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

MAIN_PATH = "/work/work_fran/Deep_Var_BCD/"
MODEL_NAME = f"{args.mnet_name}_{args.pretraining_epochs}pe_{args.patch_size}ps_{args.theta_val}theta_{args.sigmaRui_sq}sigmaRui_{args.n_samples}nsamples"
WEIGHTS_PATH = args.weights_path + f"{MODEL_NAME}/"

SAVE_PATH = args.save_path + f"BCDNET_{MODEL_NAME}/"

dataloader = get_files_dataloader(args.dataset_path, None, args.batch_size, args.num_workers)

model = DVBCDModel(mnet_name = args.mnet_name, cnet_name = args.cnet_name, device=DEVICE)
model.load(WEIGHTS_PATH + "best.pt", remove_module=False)
model.eval()

print("Deconvolving...")
filenames_list = []
out_Mnet_mean_list = []
out_Cnet_list = []
for idx, batch in enumerate(tqdm(dataloader)):
    Y, Y_OD, filenames = batch
    Y_OD = Y_OD.to(DEVICE)
    out_Mnet_mean, _, out_Cnet, _ = model.forward(Y_OD)
    out_Mnet_mean = out_Mnet_mean.cpu().detach().numpy()
    out_Cnet = out_Cnet.cpu().detach().numpy()

    out_Mnet_mean_list = out_Mnet_mean_list + list(out_Mnet_mean)
    out_Cnet_list = out_Cnet_list + list(out_Cnet)
    filenames_list = filenames_list + list(filenames)

print("Saving files...")
for idx in tqdm(range(len(filenames_list))):
    filename = filenames_list[idx]
    new_file = filename.split(args.dataset_path)[1]
    new_file = new_file.split('.')[0] + ".Results.mat"
    new_file = SAVE_PATH + new_file

    out_Mnet_mean = out_Mnet_mean_list[idx]
    out_Cnet = out_Cnet_list[idx]

    if not os.path.exists(os.path.dirname(new_file)):
        os.makedirs(os.path.dirname(new_file))

    h5f = h5py.File(new_file, 'w')
    h5f.create_dataset('M', data=out_Mnet_mean)
    h5f.create_dataset('stains', data=out_Cnet)
    h5f.close()