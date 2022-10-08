import argparse
import random


def set_opts():
    parser = argparse.ArgumentParser()

    # trainning settings
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size of training, (default:4)")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch size of data sample,  (default:256)")
    parser.add_argument('--epochs', type=int, default=100, help="Training epohcs")
    parser.add_argument('-p', '--print_freq', type=int, default=10, help="Print frequence (default: 100)")
    parser.add_argument('-s', '--save_model_freq', type=int, default=10, help="Save model frequence (default: 10)")

    # learning rate
    parser.add_argument('--lr_D', type=float, default=1e-6, help="Initialized learning rate of DNet, (default: 1e-4)")
    parser.add_argument('--lr_K', type=float, default=1e-4, help="Initialized learning rate of KNet, (default: 1e-4)")
    parser.add_argument('--gamma', type=float, default=0.1, help="Decaying rate for the learning rate, (default: 0.1)")

    # Cliping the Gradients Norm during the training
    parser.add_argument('--clip_grad_D', type=float, default=1e4, help="Cliping the gradients for D-Net, (default: 1e4)")
    parser.add_argument('--clip_grad_K', type=float, default=1e5, help="Cliping the gradients for K-Net, (default: 1e5)")

    parser.add_argument('--data_center_path', default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', type=str, metavar='PATH',
                        help="Path to save the GoPro dataset images")
    parser.add_argument('--data_center_test_path',
                        default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', type=str,
                        metavar='PATH',
                        help="Path to save the GoPro dataset images")
    parser.add_argument('--pre_kernel_path', default='', type=str, metavar='PATH',
                        help="Path to save the Pre Generate kernels.")

    # model and log saving
    parser.add_argument('--log_dir', default='./log_gt', type=str, metavar='PATH', help="Path to save the log file, (default: ./log)")
    parser.add_argument('--model_dir', default='./model_gt', type=str, metavar='PATH', help="Path to save the model file, (default: ./model)")
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help="Path to the latest checkpoint (default: None)")
    parser.add_argument('--num_workers', default=8, type=int, help="Number of workers to load data, (default: 8)")

    # hyper-parameters
    parser.add_argument('--eps2', default=1e-4, type=float,
                        help="Variance for prior of Z, (default: 1e-6)")
    parser.add_argument('--alpha0', default=1e-3, type=float, help="Variance for prior of p(x), (default: 1e-6)")
    parser.add_argument('--alpha_e2', default=1e-3, type=float, help="Variance for prior of p(x), (default: 1e-6)")
    parser.add_argument('--alpha_h2', default=1e-3, type=float, help="Variance for prior of p(x), (default: 1e-6)")

    parser.add_argument('--sigma2', default=1e-1, type=float, help="Variance for p(y|k, x), (default: 1e-5)")


    # network architecture
    parser.add_argument('--code_len', type=int, default=30, help="kernel code len")
    parser.add_argument('--DNet', type=str, default='unet_6', help='Deblur Net Model')
    parser.add_argument('--KNet', type=str, default='resnet_18_in', help='kernel net')


    parser.add_argument('--pretraining_epoch', type=int, default=0, help='Pretraining for kd mode')

    # others
    parser.add_argument('--nrow', type=int, default=8, help="make grid nrow")
    parser.add_argument('--epoch_start_test', type=int, default=1, help="epoch_start_test")
    parser.add_argument('--skip_grad', type=float, default=1e6, help="Skip this batch")



    parser.add_argument('--run_mode', type=str, default="center_0", help='Train which dataset, one of ["center_0", "center_1", "center_2", "center_3"]')

    # multi-gpu setting
    parser.add_argument('--rank', default=0, type=int, help='Node rank for distributed training, 0 - n - 1')
    parser.add_argument('--n_node', default=1, type=int, help='Number of node.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:34567', type=str,
                        help='url used to set up distributed training. This should be'
                             'the IP address and open port number of the master node')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')

    args = parser.parse_args()
    return args
