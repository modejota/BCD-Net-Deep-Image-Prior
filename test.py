import os
import torch

from options import set_test_opts

from utils.utils_data import get_test_dataloaders
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

args = set_test_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

test_dataloader_camelyon, test_dataloader_wssb_dict = get_test_dataloaders(args.test_data_path, args.patch_size, args.batch_size, args.num_workers, val_prop=args.val_prop, n_samples=args.n_samples)

model = DVBCDModel(cnet_name="unet_6", mnet_name="resnet_18_in", device=DEVICE)
model.load_model(model_path)