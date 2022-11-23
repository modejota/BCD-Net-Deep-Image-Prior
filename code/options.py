import argparse
import numpy as np


def set_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                        '--camelyon_data_path', default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument(
                        '--wssb_data_path', default='/data/BasesDeDatos/Alsubaie/Data/', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument('--save_history_dir', default='/work/work_fran/Deep_Var_BCD/history/', type=str, metavar='PATH', help="Path to save the history file (default: /work/work_fran/Deep_Var_BCD/history/)")
    parser.add_argument('--save_model_dir', default='/work/work_fran/Deep_Var_BCD/weights/', type=str, metavar='PATH', help="Path to save the model weights (default: /work/work_fran/Deep_Var_BCD/weights/)")   
    parser.add_argument('--results_dir', default='/work/work_fran/Deep_Var_BCD/results/', type=str, metavar='PATH', help="Path to save the results (default: /work/work_fran/Deep_Var_BCD/results/)")   
    parser.add_argument('--save_freq', default=10, type=int, help="Frequency to save the model weights (default: 10)")

    # trainning settings
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size of training (default: 32)")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size of data sample  (default: 224)")
    parser.add_argument('--val_prop', type=float, default=0.1, help="Proportion of validation data (default: 0.1)")
    parser.add_argument('--epochs', type=int, default=20, help="Training epochs (default: 40)")
    parser.add_argument('--pretraining_epochs', type=int, default=1, help="Pretraining epohcs (default: 1)")
    parser.add_argument('--n_samples', type=int, default=60000, help="Number of samples to use for training (default: 60000)")

    parser.add_argument('--num_workers', default=16, type=int, help="Number of workers to load data, (default: 16)")
    parser.add_argument('--resume_path', default='', type=str, metavar='PATH', help="Path to the latest checkpoint (default: None)")

    # learning rate
    parser.add_argument('--lr_cnet', type=float, default=1e-4, help="Initial learning rate of CNet (default: 1e-6)")
    parser.add_argument('--lr_mnet', type=float, default=1e-4, help="Initial learning rate of MNet (default: 1e-4)")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="Decaying rate for the learning rate (default: 0.5)")
    parser.add_argument('--patience', type=int, default=10, help="Patience for the learning rate scheduler (default: 10)")

    # How to clip the gradients norm during the training
    parser.add_argument('--clip_grad_cnet', type=float, default=1e4, help="Value to clip the gradients of CNet, (default: 1e4)")
    parser.add_argument('--clip_grad_mnet', type=float, default=1e5, help="Value to clip the gradients of MNet, (default: 1e5)")
   
    # hyper-parameters
    parser.add_argument('--sigmaRui_sq', default=0.05, type=float, help="Prior hematoxylin/eosin variance of M (default: 0.05)")
    parser.add_argument('--theta_val', default=0.5, type=float, help="Theta hyperparameter to balance the loss function (default: 0.5)")
    #parser.add_argument('--lambda_val', default=0.5, type=float, help="Lambda parameter to balance the loss function (default: 0.5)")


    args = parser.parse_args()
    return args