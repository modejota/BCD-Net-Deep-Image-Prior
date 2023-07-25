from options import set_opts
args = set_opts()

import os
import sys
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']=args.device[-1] if args.device != 'cpu' else ''
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import numpy as np

from models import Cnet, BCDnet
from datasets import WSSBDatasetTest
from utils import od2rgb, rgb2od, random_ruifrok_matrix_variation, direct_deconvolution, peak_signal_noise_ratio, structural_similarity, askforPyTorchWeightsviaGUI

SAVE_WEIGHTS = args.save_weights
RUN_FROM_WEIGHTS = args.load_weights

APPROACH_USED = args.approach
BATCH_SIZE = 1  # Should always be 1
SIGMA_RUI_SQ =  args.sigma_rui_sq
LEARNING_RATE = args.lr

ORGAN_LIST = set(args.organs)
NUM_ITERATIONS = args.iterations
# Always running on Delfos

torch.manual_seed(0)
device = torch.device(args.device[:-2])
print('Using device:', device)

alsubaie_dataset_path = args.wssb_data_path

metrics_dict = {
    'epoch' : 0, 'loss': 0.0, 'time': 0.0,
    'mse_rec' : 0.0, 'psnr_rec': 0.0, 'ssim_rec': 0.0,
    'mse_gt_h': 0.0, 'mse_gt_e': 0.0, 'mse_gt': 0.0,
    'psnr_gt_h': 0.0, 'psnr_gt_e': 0.0, 'psnr_gt': 0.0,
    'ssim_gt_h': 0.0, 'ssim_gt_e': 0.0, 'ssim_gt': 0.0
}


# Create the folder for the results
folder_route = f'../../results/{APPROACH_USED}/batch_training'
if not os.path.exists(folder_route):
    os.makedirs(folder_route)

if 'bcdnet' in APPROACH_USED and RUN_FROM_WEIGHTS:
    pretrained_weights_filepath = askforPyTorchWeightsviaGUI()

for organ in ORGAN_LIST:
    # One dataset for each organ would help to keep track of the results
    dataset = WSSBDatasetTest(alsubaie_dataset_path, organ_list=[organ], load_at_init=False)

    # Train the model and evaluate for each image
    for index, (image, M_gt, _) in enumerate(dataset):

        print(f"Organ: {organ} \t Image: {index}")

        if 'bcdnet' in APPROACH_USED:
            model = BCDnet(cnet_name='unet_64_6', mnet_name='mobilenetv3s_50').to(device)
        elif 'cnet' in APPROACH_USED:
            model = Cnet().to(device)
        else:
            raise Exception('Approach not found.')

        # We only have general pretrained weights for BCDNET. If want to fine-tune for an specific image, use the other script.
        if RUN_FROM_WEIGHTS:
            model.load_state_dict(torch.load(pretrained_weights_filepath))

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        input_noise = torch.rand(image.shape).unsqueeze(0).to(device)  # Unsqueezed to add the batch dimension, it needs to be ([1, 3, x, y])
        original_tensor = image.unsqueeze(0).to(device)
        _, _, H, W = original_tensor.shape
        M_gt = M_gt.to(device)

        # Create the metrics file and fill the header
        metrics_filepath = folder_route + f"/metrics/{organ}_{index}.csv"
        with(open(metrics_filepath, 'w')) as file:
            header = ','.join(metrics_dict.keys()) + '\n'
            file.write(header)

        for iteration in range(NUM_ITERATIONS):

            start_time = time.time()
            optimizer.zero_grad()

            if APPROACH_USED in ['bcdnet_e1', 'bcdnet_e2']:
                # Using BCDnet we obtain both the concentration matrix and the colors matrix as well as the colors' variation
                M_matrix, M_variation, C_matrix = model(input_noise)

            elif APPROACH_USED == 'cnet_e2':
                # Using Cnet we just obtain the concentration matrix
                C_matrix = model(input_noise)

                # Generate the colors matrix as a sample of a gaussian distribution given the Ruifrok matrix
                h_matrix, e_matrix = random_ruifrok_matrix_variation(args.sigma_rui_sq)
                M_matrix = np.concatenate((h_matrix,e_matrix),axis=1)                   # ([1, 3, 2])
                M_matrix = torch.from_numpy(M_matrix).float().unsqueeze(0).to(device)   # ([1, 2, x, y])

            # Generate the 3 channels image and get it back to RGB
            reconstructed_od = torch.einsum('bcs,bshw->bchw', M_matrix, C_matrix)   # ([1, 3, x, y])
            reconstructed = torch.clamp(od2rgb(reconstructed_od), 0, 255.0)

            if APPROACH_USED in ['bcdnet_e1', 'cnet_e2']:
                loss = torch.nn.functional.mse_loss(reconstructed, original_tensor)

            elif APPROACH_USED == 'bcdnet_e2':
                ruifrok_matrix = torch.tensor([
                                    [0.6442, 0.0928],
                                    [0.7166, 0.9541],
                                    [0.2668, 0.2831]
                                    ]).type(torch.float32)
                ruifrok_matrix = ruifrok_matrix.repeat(BATCH_SIZE, 1, 1).to(device)  # (batch_size, 3, 2)
                M_variation = M_variation.repeat(1, 3, 1)   # (batch_size, 3, 2)
                # Calculate the Kullback-Leiber divergence via its closed form
                loss_kl = (0.5 / SIGMA_RUI_SQ) * torch.nn.functional.mse_loss(M_matrix, ruifrok_matrix, reduction='none') + 1.5 * (M_variation / SIGMA_RUI_SQ - torch.log(M_variation / SIGMA_RUI_SQ) - 1) # (batch_size, 3, 2)
                loss_kl = torch.sum(loss_kl) / BATCH_SIZE # (1)
                loss_mse = torch.nn.functional.mse_loss(reconstructed, original_tensor)
                loss = loss_mse + loss_kl

            loss.backward()
            optimizer.step()

            # Calculate general metrics and reconstruction metrics
            metrics_dict['time'] = ((time.time() - start_time) * 1000.0)  # Milliseconds
            metrics_dict['epoch'] = iteration+1
            metrics_dict['loss'] = loss.item()
            metrics_dict['mse_rec'] = torch.sum(torch.pow(reconstructed_od - rgb2od(original_tensor), 2)).item() / (3.0*H*W)
            metrics_dict['psnr_rec'] = torch.sum(peak_signal_noise_ratio(reconstructed, original_tensor)).item()
            metrics_dict['ssim_rec'] = torch.sum(structural_similarity(reconstructed, original_tensor)).item()

            # Generate the ground truth images
            C_gt = direct_deconvolution(rgb2od(image), M_gt).unsqueeze(0) # (1, 2, H, W)

            gt_od = torch.einsum('bcs,bshw->bschw', M_gt, C_gt)
            H_gt_od = gt_od[:,0,:,:]
            H_gt = torch.clamp(od2rgb(H_gt_od), 0.0, 255.0) # (batch_size, 3, H, W)
            E_gt_od = gt_od[:,1,:,:]
            E_gt = torch.clamp(od2rgb(E_gt_od), 0.0, 255.0) # (batch_size, 3, H, W)

            H_gt = H_gt.to(device)
            H_gt_od = H_gt_od.to(device)
            E_gt = E_gt.to(device)
            E_gt_od = E_gt_od.to(device)

            # Generate the images from the model
            C_mean = C_matrix.detach().cpu()

            rec_od = torch.einsum('bcs,bshw->bschw', M_matrix, C_matrix)
            H_rec_od = rec_od[:,0,:,:]
            H_rec = torch.clamp(od2rgb(H_rec_od), 0.0, 255.0)
            E_rec_od = rec_od[:,1,:,:]
            E_rec = torch.clamp(od2rgb(E_rec_od), 0.0, 255.0)

            # Calculate the metrics comparing with the ground truth
            metrics_dict['mse_gt_h'] = torch.sum(torch.pow(H_gt_od - H_rec_od, 2)).item() / (3.0*H*W)
            metrics_dict['mse_gt_e'] = torch.sum(torch.pow(E_gt_od - E_rec_od, 2)).item() / (3.0*H*W)

            metrics_dict['psnr_gt_h'] = torch.sum(peak_signal_noise_ratio(H_gt, H_rec)).item()
            metrics_dict['psnr_gt_e'] = torch.sum(peak_signal_noise_ratio(E_gt, E_rec)).item()

            metrics_dict['ssim_gt_h'] = torch.sum(structural_similarity(H_gt, H_rec)).item()
            metrics_dict['ssim_gt_e'] = torch.sum(structural_similarity(E_gt, E_rec)).item()

            metrics_dict['mse_gt'] = (metrics_dict['mse_gt_h'] + metrics_dict['mse_gt_e'])/2.0
            metrics_dict['psnr_gt'] = (metrics_dict['psnr_gt_h'] + metrics_dict['psnr_gt_e'])/2.0
            metrics_dict['ssim_gt'] = (metrics_dict['ssim_gt_h'] + metrics_dict['ssim_gt_e'])/2.0

            with open(metrics_filepath, mode='a') as file:
                data_row = ','.join(str(val) for val in metrics_dict.values()) + '\n'
                file.write(data_row)

        # Save weights at the end in case we want to train further from this point.
        if SAVE_WEIGHTS:
            save_weights_filepath = folder_route + f"/weights/{organ}_{index}_iteration_{NUM_ITERATIONS}.pt"
            if os.path.exists(save_weights_filepath):
                new_number_iterations = int(save_weights_filepath.split('_')[-1].split('.')[0]) + NUM_ITERATIONS
                save_weights_filepath = folder_route + f"/weights/{organ}_{index}_iteration_{new_number_iterations}.pt"
            torch.save(model.state_dict(), save_weights_filepath)

    # Trying to reduce wasted memory
    torch.cuda.empty_cache()
