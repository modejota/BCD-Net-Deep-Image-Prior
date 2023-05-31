import os
import torch

from datasets import CamelyonDataset, WSSBDatasetTest
from models.DVBCDModel import DVBCDModel
from callbacks import EarlyStopping, ModelCheckpoint, History

def build_model(args):
    model = DVBCDModel(
                    cnet_name=args.cnet_name, mnet_name=args.mnet_name, 
                    sigmaRui_sq=args.sigma_sq, theta_val=args.theta_val, 
                    lr=args.lr, lr_decay=args.lr_decay, clip_grad=args.clip_grad
                    )


def train(args, wandb_run=None):


    save_model_path = os.path.join(args.save_model_dir, f"{args.save_model_name}/")
    history_path = os.path.join(args.save_history_dir, f"{args.save_model_name}.csv")
    TRAIN_CENTERS =  [0,2,4]
    VAL_CENTERS = [1,3]

    if not os.path.exists(args.save_history_dir):
        os.makedirs(args.save_history_dir)

    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

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

    callbacks = [
        EarlyStopping(model, score_name=ES_METRIC, mode="min", delta=0.0, patience=args.patience, path=save_model_path), 
        ModelCheckpoint(model, path=save_model_path, save_freq=args.save_freq), 
        History(path = history_path, wandb_run=wandb_run)]
    model.set_callbacks(callbacks)

    n_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params}")

    print("Using", torch.cuda.device_count(), "GPUs!")
    model.DP()
    model.to(args.device)

    if args.pretraining_epochs > 0:
        model.theta_val = 0.999
        model.fit(args.pretraining_epochs, train_dataloader, val_dataloader)
        model.init_optimizer()
    model.theta_val = args.theta_val
    model.fit(args.epochs, train_dataloader, val_dataloader)

def test(args):
    pass