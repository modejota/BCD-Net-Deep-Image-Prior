import argparse

def set_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help="Device to use for training")

    parser.add_argument('--sgti', action='store_true', help="Save ground truth images. Only for individual image processing")
    parser.add_argument('--smgi', action='store_true', help="Save model generated images during training. Only for individual image processing")
    parser.add_argument('--smgi_frequency', type=int, default=200, help="Interval between saving images (in iterations) generated by the model")
    parser.add_argument('--save_weights', action='store_true', help="Save model weights at the end of training training")
    parser.add_argument('--load_weights', action='store_true', help="Load model weights at the beginning of training")

    parser.add_argument('--approach', choices=['bcdnet_e1', 'bcdnet_e2', 'bcdnet_e3', 'cnet_e2'], help="Approach used during training")
    parser.add_argument('--organ', choices=['Lung', 'Breast', 'Colon'], default='Colon', help="Organ used during training. Only for individual processing")
    parser.add_argument('--organs', nargs='+', choices=['Lung', 'Breast', 'Colon'], help='Organs used during training.')
    parser.add_argument('--image_id', type=int, help="Image ID to process. Only for individual processing", default=0)

    parser.add_argument(
                        '--load_weights_path', default='/home/modej/Deep_Var_BCD/BCDNET_pretrained_weights.pt',
                        type=str, metavar='PATH', help="Path to load the model weights"
                        )
    parser.add_argument(
                        '--wssb_data_path', default='/home/modej/Alsubaie_500x500',
                        type=str, metavar='PATH', help="Path to load the WSSB dataset images"
                        )

    parser.add_argument('--lr', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--sigma_rui_sq', default=0.05, type=float, help="Prior hematoxylin/eosin variance of M")
    parser.add_argument('--iterations', type=int, default=4000, help="Number of training iterations")
    parser.add_argument('--coloriter', type=int, default=1000, help="Number of iterations focusing on getting the correct colors. Only in BCDNET_E3")
    parser.add_argument('--theta_val', type=float, default=0.5, help="Value of theta (ponderation between rec and kl losses). (1-theta)*rec_loss + theta*kl_loss.")
    parser.add_argument('--theta_val_coloriter', type=float, default=0.99, help="Value of theta (ponderation between rec and kl losses) for the coloriter iterations. (1-theta)*rec_loss + theta*kl_loss. Only in BCDNET_E3.")

    args = parser.parse_args()
    return args