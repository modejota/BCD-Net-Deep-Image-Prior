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
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.BCDnet import BCDnet
from models.cnet import Cnet
from datasets import WSSBDatasetTest
from utils import od2rgb, rgb2od, random_ruifrok_matrix_variation, direct_deconvolution, peak_signal_noise_ratio, structural_similarity

SAVE_GROUND_TRUTH_IMAGES = args.sgti
SAVE_MODEL_GENERATED_IMAGES = args.smgi
SAVE_IMAGES_FREQUENCY = args.smgi_frequency
SAVE_WEIGHTS = args.save_weights
RUN_FROM_WEIGHTS = args.load_weights
START_FROM_IMAGE_ITSELF = args.use_image_itself

APPROACH_USED = args.approach
BATCH_SIZE = 1  # Should always be 1
SIGMA_RUI_SQ = args.sigma_rui_sq
LEARNING_RATE = args.lr
THETA_VAL = args.theta_val
THETA_VAL_COLORITER = args.theta_val_coloriter
COLORITER = args.coloriter

ORGAN = args.organ
IMAGE_TO_LOAD = args.image_id

torch.manual_seed(0)
plt.rcParams['font.size'] = 14
plt.rcParams['toolbar'] = 'None'

device = torch.device(args.device[:-2] if args.device != 'cpu' else 'cpu')
if device == 'cuda':    # Try to improve speed as images are always the same size
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
print('Using device:', device)

alsubaie_dataset_path = args.wssb_data_path

dataset = WSSBDatasetTest(alsubaie_dataset_path, organ_list=[ORGAN], load_at_init=False)
original_image, M_gt = dataset[IMAGE_TO_LOAD]
print('Image shape:', original_image.shape)

NUM_ITERATIONS = args.iterations

metrics_dict = { 
    'epoch' : 0, 'loss': 0.0, 
    'mse_rec' : 0.0, 'psnr_rec': 0.0, 'ssim_rec': 0.0,
    'mse_gt_h': 0.0, 'mse_gt_e': 0.0, 'mse_gt': 0.0,
    'psnr_gt_h': 0.0, 'psnr_gt_e': 0.0, 'psnr_gt': 0.0,
    'ssim_gt_h': 0.0, 'ssim_gt_e': 0.0, 'ssim_gt': 0.0, 'time': 0.0    
}

if APPROACH_USED in ['bcdnet_e2', 'bcdnet_e3', 'bcdnet_e4']:
    metrics_dict['loss_rec'] = 0.0
    metrics_dict['loss_kl'] = 0.0
    if APPROACH_USED == 'bcdnet_e4':
        metrics_dict['loss_l2'] = 0.0

folder_route = f'../../results/{APPROACH_USED}/per_image_training/{ORGAN}_{IMAGE_TO_LOAD}'
if not os.path.exists(folder_route):
    os.makedirs(folder_route)

if not os.path.exists(f'{folder_route}/images') and (SAVE_MODEL_GENERATED_IMAGES or SAVE_GROUND_TRUTH_IMAGES):
    os.makedirs(f'{folder_route}/images')

metrics_filepath = f'{folder_route}/metrics.csv'
if os.path.exists(metrics_filepath):
    os.remove(metrics_filepath)

with(open(metrics_filepath, 'w')) as file:
    header = ','.join(metrics_dict.keys()) + '\n'
    file.write(header)

ruifrok_matrix = torch.tensor([
                    [0.6442, 0.0928],
                    [0.7166, 0.9541],
                    [0.2668, 0.2831]
                    ]).type(torch.float32)
ruifrok_matrix = ruifrok_matrix.repeat(BATCH_SIZE, 1, 1).to(device)  # (batch_size, 3, 2)

# Generate all images derivated from the ground truth.
img_np = original_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
img_od = rgb2od(original_image)

C_gt = direct_deconvolution(img_od, M_gt).unsqueeze(0) # (1, 2, H, W)
M_gt = M_gt.unsqueeze(0) # (1, 3, 2)

C_H_gt_np = C_gt[:, 0, :, :].squeeze().numpy() # (H, W)
C_E_gt_np = C_gt[:, 1, :, :].squeeze().numpy() # (H, W)

gt_od = torch.einsum('bcs,bshw->bschw', M_gt, C_gt)
H_gt_od = gt_od[:,0,:,:]
H_gt = torch.clamp(od2rgb(H_gt_od), 0.0, 255.0) # (batch_size, 3, H, W)
E_gt_od = gt_od[:,1,:,:]
E_gt = torch.clamp(od2rgb(E_gt_od), 0.0, 255.0) # (batch_size, 3, H, W)

H_gt_np = H_gt.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
E_gt_np = E_gt.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')

if SAVE_GROUND_TRUTH_IMAGES:
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    ax[0].imshow(img_np)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(C_H_gt_np, cmap='gray')
    ax[1].set_title('Original Hematoxylin\nConcentration')
    ax[1].axis('off')

    ax[2].imshow(C_E_gt_np, cmap='gray')
    ax[2].set_title('Original Eosin\nConcentration')
    ax[2].axis('off')

    ax[3].imshow(H_gt_np)
    ax[3].set_title('Original Hematoxylin')
    ax[3].axis('off')

    ax[4].imshow(E_gt_np)
    ax[4].set_title('Original Eosin')
    ax[4].axis('off')

    if not os.path.exists(f'{folder_route}/images'):
        os.makedirs(f'{folder_route}/images')
    plt.savefig(f'{folder_route}/images/ground_truth_image.png', transparent=True)
    plt.close()

# Move to GPU, we gonna need it later during traning for metrics' calculation.
H_gt = H_gt.to(device)
H_gt_od = H_gt_od.to(device)
E_gt = E_gt.to(device)
E_gt_od = E_gt_od.to(device)

if START_FROM_IMAGE_ITSELF:
    input = original_image.unsqueeze(0).to(device)
else:
    input = torch.rand(original_image.shape).unsqueeze(0).to(device)  # Unsqueezed to add the batch dimension, it needs to be ([1, 3, x, y])

original_tensor = original_image.unsqueeze(0).to(device)
original_tensor_od = rgb2od(original_tensor).to(device)
_, _, H, W = original_tensor.shape

if 'bcdnet' in APPROACH_USED:
    model = BCDnet(cnet_name='unet_64_6', mnet_name='mobilenetv3s_50').to(device)
elif 'cnet' in APPROACH_USED:
    model = Cnet().to(device)
else:
    raise Exception('Approach not found.')


if RUN_FROM_WEIGHTS and APPROACH_USED != 'cnet2':
    weightsfile = args.load_weights_path
    if os.path.isfile(weightsfile):
        print(f'Loading weights from {weightsfile}')
    else:
        raise Exception(f'Weights file {weightsfile} not found.')
    model.load_state_dict(torch.load(weightsfile))

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

loop_data = range(1, NUM_ITERATIONS+1)
for iteration in tqdm(loop_data, desc="Processing image", unit="item"):

    start_time = time.time()
    optimizer.zero_grad()

    if 'bcdnet' in APPROACH_USED:
        # Using BCDnet we obtain both the concentration matrix and the colors matrix as well as the colors' variation
        M_matrix, M_variation, C_matrix = model(input)        

    elif APPROACH_USED == 'cnet_e2':
        # Using Cnet we just obtain the concentration matrix
        C_matrix = model(input)

        # Generate the colors matrix as a sample of a gaussian distribution given the Ruifrok matrix
        h_matrix, e_matrix = random_ruifrok_matrix_variation(0.05)
        M_matrix = np.concatenate((h_matrix,e_matrix),axis=1)                   # ([1, 3, 2])
        M_matrix = torch.from_numpy(M_matrix).float().unsqueeze(0).to(device)   # ([1, 2, x, y])

    # Generate the 3 channels image and get it back to RGB
    reconstructed_od = torch.einsum('bcs,bshw->bchw', M_matrix, C_matrix)   # ([1, 3, x, y])
    reconstructed = torch.clamp(od2rgb(reconstructed_od), 0, 255.0)

    if APPROACH_USED in ['bcdnet_e1', 'cnet_e2']:
        loss = torch.nn.functional.mse_loss(reconstructed_od, original_tensor_od)

    elif APPROACH_USED == 'bcdnet_e2':
        M_variation = M_variation.repeat(1, 3, 1)   # (batch_size, 3, 2)
        # Calculate the Kullback-Leiber divergence via its closed form
        loss_kl = (0.5 / SIGMA_RUI_SQ) * torch.nn.functional.mse_loss(M_matrix, ruifrok_matrix, reduction='none') + 1.5 * (M_variation / SIGMA_RUI_SQ - torch.log(M_variation / SIGMA_RUI_SQ) - 1) # (batch_size, 3, 2)
        loss_kl = torch.sum(loss_kl) / BATCH_SIZE # (1)
        # Re-parametrization trick to sample from the gaussian distribution 
        M_sample = M_matrix + torch.sqrt(M_variation) * torch.randn_like(M_matrix) # (batch_size, 3, 2)

        Y_rec = torch.einsum('bcs,bshw->bchw', M_sample, C_matrix) # (batch_size, 3, H, W)
        loss_rec = torch.sum(torch.nn.functional.mse_loss(Y_rec, original_tensor_od)) / BATCH_SIZE # (1) 

        loss = (1.0 - THETA_VAL)*loss_rec + THETA_VAL*loss_kl
        
        metrics_dict['loss_rec'] = (1.0 - THETA_VAL)*loss_rec.item()
        metrics_dict['loss_kl'] = THETA_VAL*loss_kl.item()

    elif APPROACH_USED == 'bcdnet_e3':
        M_variation = M_variation.repeat(1, 3, 1)   # (batch_size, 3, 2)
        # Calculate the Kullback-Leiber divergence via its closed form
        loss_kl = (0.5 / SIGMA_RUI_SQ) * torch.nn.functional.mse_loss(M_matrix, ruifrok_matrix, reduction='none') + 1.5 * (M_variation / SIGMA_RUI_SQ - torch.log(M_variation / SIGMA_RUI_SQ) - 1) # (batch_size, 3, 2)
        loss_kl = torch.sum(loss_kl) / BATCH_SIZE # (1)
        # Re-parametrization trick to sample from the gaussian distribution
        M_sample = M_matrix + torch.sqrt(M_variation) * torch.randn_like(M_matrix) # (batch_size, 3, 2)

        Y_rec = torch.einsum('bcs,bshw->bchw', M_sample, C_matrix) # (batch_size, 3, H, W)
        loss_rec = torch.sum(torch.nn.functional.mse_loss(Y_rec, original_tensor_od)) / BATCH_SIZE # (1)
        
        if iteration < COLORITER:
            loss = (1.0 - THETA_VAL_COLORITER)*loss_rec + THETA_VAL_COLORITER*loss_kl
            metrics_dict['loss_rec'] = (1.0 - THETA_VAL_COLORITER)*loss_rec.item()
            metrics_dict['loss_kl'] = THETA_VAL_COLORITER*loss_kl.item()

        else:
            loss = (1.0 - THETA_VAL)*loss_rec + THETA_VAL*loss_kl
            metrics_dict['loss_rec'] = (1.0 - THETA_VAL)*loss_rec.item()
            metrics_dict['loss_kl'] = THETA_VAL*loss_kl.item()

    elif APPROACH_USED == 'bcdnet_e4':
        l2_norms = torch.norm(C_matrix.view(1, 2, -1), p=2, dim=1)
        l2_norms = l2_norms.view(1, 500, 500)
        l2_divided = torch.sum(l2_norms).item() / (H*W)

        M_variation = M_variation.repeat(1, 3, 1)   # (batch_size, 3, 2)
        # Calculate the Kullback-Leiber divergence via its closed form
        loss_kl = (0.5 / SIGMA_RUI_SQ) * torch.nn.functional.mse_loss(M_matrix, ruifrok_matrix, reduction='none') + 1.5 * (M_variation / SIGMA_RUI_SQ - torch.log(M_variation / SIGMA_RUI_SQ) - 1) # (batch_size, 3, 2)
        loss_kl = torch.sum(loss_kl) / BATCH_SIZE # (1)
        # Re-parametrization trick to sample from the gaussian distribution
        M_sample = M_matrix + torch.sqrt(M_variation) * torch.randn_like(M_matrix) # (batch_size, 3, 2)

        Y_rec = torch.einsum('bcs,bshw->bchw', M_sample, C_matrix) # (batch_size, 3, H, W)
        loss_rec = torch.sum(torch.nn.functional.mse_loss(Y_rec, original_tensor_od)) / BATCH_SIZE # (1)

        loss = loss_rec + l2_divided + loss_kl

        metrics_dict['loss_rec'] = loss_rec.item()
        metrics_dict['loss_kl'] = loss_kl.item()
        metrics_dict['loss_l2'] = l2_divided


    loss.backward()
    optimizer.step()

    # Calculate general metrics and reconstruction metrics
    metrics_dict['time'] = ((time.time() - start_time) * 1000.0)  # Milliseconds
    metrics_dict['epoch'] = iteration
    metrics_dict['loss'] = loss.item()
    metrics_dict['mse_rec'] = torch.sum(torch.pow(reconstructed_od - original_tensor_od, 2)).item() / (3.0*H*W)
    metrics_dict['psnr_rec'] = torch.sum(peak_signal_noise_ratio(reconstructed, original_tensor)).item()
    metrics_dict['ssim_rec'] = torch.sum(structural_similarity(reconstructed, original_tensor)).item()

    # Generate the images from the model
    C_mean = C_matrix.detach().cpu()
    img_rec_np = reconstructed.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')

    C_H_rec_np = C_mean[:, 0, :, :].squeeze().numpy()
    C_E_rec_np = C_mean[:, 1, :, :].squeeze().numpy()

    rec_od = torch.einsum('bcs,bshw->bschw', M_matrix, C_matrix)
    H_rec_od = rec_od[:,0,:,:]
    H_rec = torch.clamp(od2rgb(H_rec_od), 0.0, 255.0)
    E_rec_od = rec_od[:,1,:,:]
    E_rec = torch.clamp(od2rgb(E_rec_od), 0.0, 255.0)

    H_rec_np = H_rec.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
    E_rec_np = E_rec.squeeze().detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')

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

    # Save the metrics in a csv file
    with open(metrics_filepath, mode='a') as file:
        data_row = ','.join(str(val) for val in metrics_dict.values()) + '\n'
        file.write(data_row)

    if SAVE_MODEL_GENERATED_IMAGES and (iteration % SAVE_IMAGES_FREQUENCY == 0 or iteration == NUM_ITERATIONS or iteration == 1):
        # Plot the generated images via the model
        fig, ax = plt.subplots(1, 5, figsize=(20, 5))
        ax[0].imshow(img_rec_np)
        ax[0].set_title('Reconstructed Image')
        ax[0].axis('off')

        ax[1].imshow(C_H_rec_np, cmap='gray')
        ax[1].set_title('Reconstructed Hematoxylin\nConcentration')
        ax[1].axis('off')

        ax[2].imshow(C_E_rec_np, cmap='gray')
        ax[2].set_title('Reconstructed Eosin\nConcentration')
        ax[2].axis('off')

        ax[3].imshow(H_rec_np)
        ax[3].set_title('Reconstructed Hematoxylin')
        ax[3].axis('off')

        ax[4].imshow(E_rec_np)
        ax[4].set_title('Reconstructed Eosin')
        ax[4].axis('off')

        plt.savefig(f'{folder_route}/images/iteration_{iteration}.png', transparent=True)
        plt.close()

# Save weights at the end in case we want to train further from this point.
# No custom filepath can be specified from args for the sake of simplicity and to avoid overwriting weights.
if SAVE_WEIGHTS:
    save_weights_filepath = f'{folder_route}/iteration_{NUM_ITERATIONS}.pt'
    if os.path.exists(save_weights_filepath):
        new_number_iterations = int(save_weights_filepath.split('_')[-1].split('.')[0]) + NUM_ITERATIONS
        save_weights_filepath = f'{folder_route}/iteration_{new_number_iterations}.pt'
    torch.save(model.state_dict(), save_weights_filepath)
