import argparse
import random


def set_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
                        '--train_data_path', default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', 
                        type=str, metavar='PATH', help="Path to load the Camelyon dataset images"
                        )
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH', help="Path to save the log file (default: ./log)")
    parser.add_argument('--weights_dir', default='./weights', type=str, metavar='PATH', help="Path to save the model weights (default: ./weights)")   


    # trainning settings
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size of training (default: 16)")
    parser.add_argument('--patch_size', type=int, default=224, help="Patch size of data sample  (default:224)")
    parser.add_argument('--epochs', type=int, default=100, help="Training epohcs (default: 100)")
    parser.add_argument('--pretraining_epochs', type=int, default=5, help="Pretraining epohcs (default: 5)")
    #parser.add_argument('-p', '--print_freq', type=int, default=20, help="Print frequence (default: 100)")
    parser.add_argument('--save_model_freq', type=int, default=20, help="Save model frequence (default: 20)")
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
    parser.add_argument('--sigmaRui_h_sq', default=1e-3, type=float, help="Prior hematoxylin variance of M (default: 1e-3)")
    parser.add_argument('--sigmaRui_e_sq', default=1e-3, type=float, help="Prior eosin variance of M (default: 1e-3)")
    parser.add_argument('--theta', default=1e-4, type=float, help="Theta parameter to balance the loss function (default: 1e-4)")
    # networks arquitectures
    parser.add_argument('--cet_name', type=str, default='unet_6', help='Name for the CNet architecture (default: unet_6)')
    parser.add_argument('--mnet_name', type=str, default='resnet_18_in', help='Name for the MNet architecture (default: resnet_18_in)')


    args = parser.parse_args()
    return args