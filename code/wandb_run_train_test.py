import torch
import wandb

from options import set_opts
from utils import train, test

print(torch.__version__)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":

    torch.cuda.empty_cache()
    args = set_opts()

    for arg in vars(args):
        print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))

    wandb_run = wandb.init(project="DVBCD")

    args.model_name = f"{args.dataset_name}/{wandb_run.name}"
    
    wandb_run.config.update(args)
    train(args, wandb_run)
    test(args, wandb_run)
    wandb_run.finish()