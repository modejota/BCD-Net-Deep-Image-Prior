import argparse
import random


def set_opts():
    parser = argparse.ArgumentParser()

    # trainning settings
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size of training, (default:4)")
    parser.add_argument('--patch_size', type=int, default=128, help="Patch size of data sample,  (default:256)")
    parser.add_argument('--epochs', type=int, default=100, help="Training epohcs")
    parser.add_argument('-p', '--print_freq', type=int, default=20, help="Print frequence (default: 100)")
    parser.add_argument('-s', '--save_model_freq', type=int, default=20, help="Save model frequence (default: 10)")

    # learning rate
    parser.add_argument('--lr_C', type=float, default=1e-4, help="Initialized learning rate of CNet, (default: 1e-4)")
    parser.add_argument('--lr_M', type=float, default=1e-4, help="Initialized learning rate of MNet, (default: 1e-4)")
    parser.add_argument('--gamma', type=float, default=0.1, help="Decaying rate for the learning rate, (default: 0.1)")

    # Cliping the Gradients Norm during the training
    parser.add_argument('--clip_grad_C', type=float, default=1e4, help="Cliping the gradients for D-Net, (default: 1e4)")
    parser.add_argument('--clip_grad_M', type=float, default=1e5, help="Cliping the gradients for K-Net, (default: 1e5)")

    # dataset path setting
    # parser.add_argument('--train_data_path', default='./data/open_val/', type=str, metavar='PATH',
    #                     help="Path to save the train images")
    # parser.add_argument('--train_data_text_path', default='./data/text_train_data/', type=str, metavar='PATH',
    #                     help="Path to save the train images")
    # parser.add_argument('--test_data_lai_path', default='./data/lai/', type=str, metavar='PATH',
    #                     help="Path to save the test images")
    # parser.add_argument('--test_data_levin_path', default='./data/levin/', type=str, metavar='PATH',
    #                     help="Path to save the test images")
    # parser.add_argument('--test_data_text_path', default='./data/text_test_data/', type=str, metavar='PATH',
    #                     help="Path to save the test text images")
    # parser.add_argument('--test_data_kohler_path', default='./data/kohler/', type=str, metavar='PATH',
    #                     help="Path to save the test  kohler images")
    parser.add_argument('--train_data_path', default='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/', type=str, metavar='PATH',
                        help="Path to save the Camelyon dataset images")
    # parser.add_argument('--data_realblur_path', default='./data/real_blur/', type=str, metavar='PATH',
    #                     help="Path to save the RealBlur dataset images")
    parser.add_argument('--pre_kernel_path', default='', type=str, metavar='PATH',
                        help="Path to save the Pre Generate kernels.")

    # model and log saving
    parser.add_argument('--log_dir', default='./log', type=str, metavar='PATH', help="Path to save the log file, (default: ./log)")
    parser.add_argument('--model_dir', default='./model', type=str, metavar='PATH', help="Path to save the model file, (default: ./model)")
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help="Path to the latest checkpoint (default: None)")
    parser.add_argument('--num_workers', default=8, type=int, help="Number of workers to load data, (default: 8)")

    # hyper-parameters
    parser.add_argument('--sigmaRui_h_sq', default=1e-3, type=float, help="") # sigma de la h
    parser.add_argument('--sigmaRui_e_sq', default=1e-3, type=float, help="") # sigma de la e

    parser.add_argument('--theta', default=1e-4, type=float, help="") #lambda


    # network architecture
    parser.add_argument('--code_len', type=int, default=30, help="kernel code len")
    parser.add_argument('--CNet', type=str, default='unet_6', help='Deblur Net Model')
    parser.add_argument('--MNet', type=str, default='resnet_18_in', help='kernel net')

    # blur kernel setting
    parser.add_argument('--max_size', type=int, default=3, help="Max size of kernel matrix")
    parser.add_argument('--dirichlet_para_stretch', type=int, default=20000, help="dirichlet paramaters * stretch")
    parser.add_argument('--prekernels', type=str, default="", help='path of prekernels')
    parser.add_argument('--pretraining_epoch', type=int, default=0, help='Pretraining for kd mode')

    # others
    parser.add_argument('--nrow', type=int, default=8, help="make grid nrow")
    parser.add_argument('--epoch_start_test', type=int, default=20, help="epoch_start_test")
    parser.add_argument('--skip_grad', type=float, default=1e6, help="Skip this batch")

    # about weight of kl dirichlet
    parser.add_argument('--warm_up_epoch', type=int, default=0, help="warm up")
    parser.add_argument('--kl_dir_weight', type=float, default=1.0, help="kl weight of dir")

    # test
    parser.add_argument('--test_load_model_path', type=str, default='', help="Trained model path")
    parser.add_argument('--test_load_deblur_model_path', type=str, default='', help="Trained model path")
    parser.add_argument('--test_load_kernel_model_path', type=str, default='', help="Trained model path")
    parser.add_argument('--test_dataset_name', type=str, default='GoPro', help='Test Dataset: GoPro, Text, RealBlur_J, RealBlur_R')
    parser.add_argument('--test_result_dir', default='', type=str, help='Directory for results')
    parser.add_argument('--test_deblur_dir', default='', type=str, help='Directory for results')

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

def set_opts_jp():

    # trainning settings

    parser = {}

    parser['batch_size'] = 2
    parser['patch_size'] = 128
    parser['epochs'] = 5
    parser['pretraining_epochs'] = 1
    parser['print_freq'] = 1
    parser['save_model_freq'] = 20

    parser["val_props"] = 0.1

    parser['lr_C'] = 1e-4
    parser['lr_M'] = 1e-4
    parser['gamma'] = 0.1

    parser['clip_grad_C'] = 1e4
    parser['clip_grad_M'] = 1e5

    parser['train_data_path'] = '/data/BasesDeDatos/Camelyon/Camelyon17/training/Toy/'
    parser['pre_kernel_path'] = ''

    parser['log_dir'] = './log'
    parser['model_dir'] = './model'
    parser['resume'] = ''
    parser['num_workers'] = 8

    parser['sigmaRui_h_sq'] = 1e-3
    parser['sigmaRui_e_sq'] = 1e-3

    parser['theta'] = 0.5
    parser['pre_kl'] = 1e2
    parser['pre_mse'] = 1e-2

    parser['code_len'] = 30
    parser['CNet'] = 'unet_6'
    parser['MNet'] = 'resnet_18_in'

    parser['max_size'] = 3
    parser['dirichlet_para_stretch'] = 20000
    parser['prekernels'] = ""

    parser['nrow'] = 8
    parser['epoch_start_test'] = 20
    parser['skip_grad'] = 1e6

    parser['warm_up_epoch'] = 0
    parser['kl_dir_weight'] = 1.0

    parser['run_mode'] = "center_0"

    args = argparse.Namespace(**parser)
    return args
