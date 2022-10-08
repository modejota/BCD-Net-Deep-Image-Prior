import argparse
import numpy as np


def set_train_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                        '--camelyon_data_path', default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument(
                        '--wssb_data_path', default='/data/BasesDeDatos/Alsubaie/Data/', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument('--log_dir', default='/work/work_fran/Deep_Var_BCD/log/', type=str, metavar='PATH', help="Path to save the log file (default: /work/work_fran/Deep_Var_BCD/log/)")
    parser.add_argument('--save_model_dir', default='/work/work_fran/Deep_Var_BCD/weights/', type=str, metavar='PATH', help="Path to save the model weights (default: /work/work_fran/Deep_Var_BCD/weights/)")   
    parser.add_argument('--results_dir', default='/work/work_fran/Deep_Var_BCD/results/', type=str, metavar='PATH', help="Path to save the results (default: /work/work_fran/Deep_Var_BCD/results/)")   
    parser.add_argument('--save_freq', default=20, type=int, help="Frequency to save the model weights (default: 20)")


    # trainning settings
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size of training (default: 16)")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size of data sample  (default: 224)")
    parser.add_argument('--val_prop', type=float, default=0.1, help="Proportion of validation data (default: 0.1)")
    parser.add_argument('--epochs', type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument('--pretraining_epochs', type=int, default=5, help="Pretraining epohcs (default: 5)")
    parser.add_argument('--n_samples', type=int, default=60000, help="Number of samples to use for training (default: 60000)")

    parser.add_argument('--num_workers', default=64, type=int, help="Number of workers to load data, (default: 64)")
    parser.add_argument('--resume_path', default='', type=str, metavar='PATH', help="Path to the latest checkpoint (default: None)")

    # learning rate
    parser.add_argument('--lr_cnet', type=float, default=1e-4, help="Initial learning rate of CNet (default: 1e-4)")
    parser.add_argument('--lr_mnet', type=float, default=1e-4, help="Initial learning rate of MNet (default: 1e-4)")
    parser.add_argument('--lr_decay', type=float, default=0.5, help="Decaying rate for the learning rate (default: 0.1)")
    parser.add_argument('--patience', type=int, default=10, help="Patience for the learning rate scheduler (default: 10)")

    # How to clip the gradients norm during the training
    parser.add_argument('--clip_grad_cnet', type=float, default=np.Inf, help="Value to clip the gradients of CNet, (default: Inf)")
    parser.add_argument('--clip_grad_mnet', type=float, default=np.Inf, help="Value to clip the gradients of MNet, (default: Inf)")
   
    # hyper-parameters
    parser.add_argument('--sigmaRui_h_sq', default=1.0e-03, type=float, help="Prior hematoxylin variance of M (default: 1e-3)")
    parser.add_argument('--sigmaRui_e_sq', default=1.0e-03, type=float, help="Prior eosin variance of M (default: 1e-3)")
    parser.add_argument('--lambda_val', default=0.5, type=float, help="Lambda parameter to balance the loss function (default: 0.5)")


    args = parser.parse_args()
    return args

def set_test_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                        '--train_data_path', default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH', help="Path to save the log file (default: ./log)")
    parser.add_argument('--save_model_dir', default='./weights', type=str, metavar='PATH', help="Path to save the model weights (default: ./weights)")   


    # trainning settings
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size of training (default: 16)")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size of data sample  (default:224)")
    parser.add_argument('--val_prop', type=float, default=0.1, help="Proportion of validation data (default: 0.1)")
    parser.add_argument('--epochs', type=int, default=50, help="Training epochs (default: 100)")
    parser.add_argument('--pretraining_epochs', type=int, default=5, help="Pretraining epohcs (default: 5)")
    parser.add_argument('--n_samples', type=int, default=None, help="Number of samples to use for training (default: None)")

    parser.add_argument('--num_workers', default=64, type=int, help="Number of workers to load data, (default: 64)")
    parser.add_argument('--resume_path', default='', type=str, metavar='PATH', help="Path to the latest checkpoint (default: None)")

    # learning rate
    parser.add_argument('--lr_cnet', type=float, default=1e-4, help="Initial learning rate of CNet (default: 1e-4)")
    parser.add_argument('--lr_mnet', type=float, default=1e-4, help="Initial learning rate of MNet (default: 1e-4)")
    parser.add_argument('--lr_decay', type=float, default=0.1, help="Decaying rate for the learning rate (default: 0.1)")

    # How to clip the gradients norm during the training
    parser.add_argument('--clip_grad_cnet', type=float, default=1e5, help="Value to clip the gradients of CNet, (default: 1e4)")
    parser.add_argument('--clip_grad_mnet', type=float, default=1e5, help="Value to clip the gradients of MNet, (default: 1e5)")
   
    # hyper-parameters
    parser.add_argument('--sigmaRui_h_sq', default=0.05, type=float, help="Prior hematoxylin variance of M (default: 0.05)")
    parser.add_argument('--sigmaRui_e_sq', default=0.05, type=float, help="Prior eosin variance of M (default: 0.05)")
    parser.add_argument('--theta', default=0.7, type=float, help="Theta parameter to balance the loss function (default: 0.7)")
    # networks arquitectures
    parser.add_argument('--cet_name', type=str, default='unet_6', help='Name for the CNet architecture (default: unet_6)')
    parser.add_argument('--mnet_name', type=str, default='resnet_18_in', help='Name for the MNet architecture (default: resnet_18_in)')


    args = parser.parse_args()
    return args