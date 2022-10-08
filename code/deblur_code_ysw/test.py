#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-05-16 16:20:01
import os
import sys
sys.path.append('./')
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TVTF
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import time
import os,sys,glob,math
from networks.cnet import get_cnet
from networks.mnet import get_mnet
from scipy.io import savemat
from options3 import set_opts
import scipy.io
import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.io import loadmat




args = set_opts()

print(torch.version.cuda)
print(torch.cuda.is_available())

use_gpu = True
# USE_GPU = True
# if USE_GPU and torch.cuda.is_available():
#     device = torch.device('cuda:0')
# else:
#     device = torch.device('cpu')
block_size=64
C = 3
dep_U=4
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# load the pretrained model
print('Loading the Model')
model_dir='./model_mse_sum_kl_test_al0_9_pre/model_100'
checkpoint = torch.load(model_dir,map_location='cpu')


def get_od_mearsurement(out_CNet, out_MNet1,out_MNet2,batch_size,patch_size):
    # M = np.concatenate((out_MNet1[:, :, 0, :], out_MNet2[:, :, 0, :]), axis=1)
    M = out_MNet1[:, 0, :, :]
    M=M[0,:,:].T
    C=out_CNet.reshape(2,patch_size*patch_size)

    od_img=np.matmul(M,C) #B,3,4096
    od_img=od_img.T
    od_img=od_img.reshape(patch_size,patch_size,3)

    return od_img

def od2rgb(od):
    rgb=256*np.exp(-od)-1
    return rgb

def rgb2od(rgb):
    od=(-np.log((rgb+1)/256))
    return od

def display_M(M):
    ns=M.shape[1]
    Mdisp=M.T.reshape(1,ns,3)
    Mdisp=od2rgb(Mdisp).astype('int')
    plt.imshow(Mdisp)
def normalize(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)

def de_normalize(data, max_val, min_val):
    return data*(max_val - min_val)+min_val


def get_range(array):
    print('min:',array.min(),'max',array.max(),array.shape)

def get_C_rgb(C,M):
    ns=C.shape[0]
    patch_size=np.sqrt(C.shape[1]).astype('int')
    all_images= np.zeros([ns,patch_size,patch_size,3],dtype='uint8')
    for i in range(ns):
        S_od= np.dot(M[:,i].reshape(3,1),C[i,:].reshape(1,-1))
        S_rgb = np.clip(od2rgb(S_od),0,255).astype('uint')
        S_rgb=S_rgb.reshape(3,patch_size,patch_size).T
        get_range(S_rgb)
        all_images[i,:,:,:] = S_rgb
    return all_images


def directDeconvolve(Y, M):
    #     I=im2uint(I)
    [m, n, c] = Y.shape
    Y = Y.reshape(-1, c).T  # 3xMN
    #     Y=rgb2od(Icol)
    CT = np.linalg.lstsq(M, Y, rcond=None)[0]
    return CT



cnet = get_cnet(args.CNet)
mnet = get_mnet(args.MNet, kernel_size=3)


if use_gpu:
    # dnet = dnet.cuda()#######load net to CUDA
    # knet = knet.cuda()
    cnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)

    mnet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)
else:
    cnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)
    mnet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)
cnet.eval()
mnet.eval()

RGB_dir= '/data/BasesDeDatos/Alsubaie/Data/RGB_images/'
GT_dir= '/data/BasesDeDatos/Alsubaie/Data/GroundTruth/'

#Cada iamgen tiene varios parches
image_patches=glob.glob(RGB_dir+'/*/*/*/*')
for image in image_patches[0:]:
    # I_GT=plt.imread(image)
    GT_file=image.replace(RGB_dir,GT_dir)
    GT_file=GT_file.rsplit('/',2)[0]
    GT_file=glob.glob(GT_file+'/*SV.mat')[0]
    M_GT=loadmat(GT_file)
    print(image)
    print(GT_file)


    # Savemat
    img = os.path.split(image)[-1]

    n = image.rfind("RGB_images")
    img_name = image[n+11:]

    im_dir = os.path.dirname(img_name)

    m_dir = os.path.join('/work/work_ysw/Deblur_Code/test_result/RGB_images_model_mse_sum_kl_test_al0_9_pre', im_dir)
    os.makedirs(m_dir, exist_ok=True)
    # save_path = os.path.join(m_dir, '%s.mat' % img)
    save_path =m_dir


    im = img_as_float(cv2.imread(image)[:, :, ::-1])
    H, W, _ = im.shape
    if H % 2 ** dep_U != 0:
        H -= H % 2 ** dep_U
    if W % 2 ** dep_U != 0:
        W -= W % 2 ** dep_U

    row = im.shape[0]
    col = im.shape[1]

    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)

    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)
    row_new = row + row_pad
    col_new = col + col_pad

    max_rol = row_new - block_size
    max_col = col_new - block_size

    im_y1 = im[row - row_pad:, col - col_pad:, :]
    im_y = np.zeros((row_new, col_new, 3), float)
    im_y[:row, :col, :] = im
    im_y[row:row_new, col:col_new, :] = im_y1

    # im_y = im[:H, :W, ]

    im_y = rgb2od(im)
    # im_y= img_as_float(im)
    # im_y  = normalize(im_y , max_val=-np.log((1)/256), min_val=0.)
    im_y = normalize(im_y, max_val=np.max(im_y), min_val=np.min(im_y))

    if use_gpu:
        im_y = torch.tensor(im_y).cuda()
        im_y = torch.permute(im_y, [2, 0, 1])
        im_y = im_y.unsqueeze(0)
        im_y = im_y.type(torch.FloatTensor)
        print('Begin Testing on GPU')
    else:
        im_y = torch.from_numpy(im_y)
        im_y = torch.permute(im_y, [2, 0, 1])
        im_y = im_y.unsqueeze(0)
        im_y = im_y.type(torch.FloatTensor)
        print('Begin Testing on CPU')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # with torch.autograd.set_grad_enabled(False):
    with torch.no_grad():
        # tic = time.perf_counter()
        starter.record()
        out_MNet1, out_MNet2 = mnet(im_y)

        im_c = cnet(im_y)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        # toc = time.perf_counter()
        im_c = im_c.cpu().numpy()

        out_MNet1 = out_MNet1.cpu().numpy()
        out_MNet2 = out_MNet2.cpu().numpy()

    mat_dir = os.path.join(save_path, 'c_epoch100.mat')
    scipy.io.savemat(mat_dir, {'c_result': im_c[0, :2, :row, :col]})

    re_y = get_od_mearsurement(im_c[:, :2, :row, :col], out_MNet1, out_MNet2, row, col)

    re_y = de_normalize(re_y, max_val=np.max(re_y), min_val=np.min(re_y))
    re_y = od2rgb(re_y)
    re_y = re_y / 255

    plt.subplot(231)
    plt.imshow(im)
    plt.title('Y Image')

    plt.subplot(232)
    plt.imshow(im_c[0, 0, :row, :col])
    plt.title('H Image')

    plt.subplot(233)
    plt.imshow(im_c[0, 1, :row, :col])
    plt.title('E Image')

    plt.subplot(234)
    plt.imshow(re_y)
    plt.title('Y_re Image')
    plt.show()

    print(f"m_h={out_MNet1[:, :, 0, :]}")
    print(f"m_e={out_MNet1[:, :, 1, :]}")
    print('time={:.4f}'.format(curr_time))

    m_out = out_MNet1[:, 0:, :]
    # #
    display_M(m_out.T)

    plt.show()

    m_h = out_MNet1[:, :, 0, :].squeeze(0)

    m_e = out_MNet1[:, :, 1, :].squeeze(0)

    sigma_h = out_MNet2[:, 0]
    sigma_e = out_MNet2[:, 1]

    M_read = np.zeros([2, 3])
    M_read[0, :] = m_h[0, :]
    M_read[1, :] = m_e[0, :]
    M_net = M_read.T

    C_norm = im_c.reshape(2, -1)

    C_net = C_norm
    [ns, w] = C_net.shape
    new_w = int(w ** 0.5)

    save_h_path = os.path.join(save_path, 'H.png')

    H_od = np.dot(M_net[:, 0].reshape(3, 1), C_net[0, :].reshape(1, -1))
    H_rgb = od2rgb(H_od).astype('int')
    H_rgb = H_rgb.T
    H_rgb = H_rgb.reshape(new_w, new_w, 3)
    H_im = np.clip(H_rgb, 0, 255)
    H_im = Image.fromarray(H_im.astype(np.uint8), mode='RGB')
    H_im.save(save_h_path)

    plt.subplot(1, ns, 0 + 1)
    plt.imshow(H_rgb)

    save_e_path = os.path.join(save_path, 'E.png')
    E_od = np.dot(M_net[:, 1].reshape(3, 1), C_net[1, :].reshape(1, -1))
    E_rgb = od2rgb(E_od).astype('int')
    E_rgb = E_rgb.T
    E_rgb = E_rgb.reshape(new_w, new_w, 3)
    E_im = np.clip(E_rgb, 0, 255)
    E_im = Image.fromarray(E_im.astype(np.uint8), mode='RGB')
    E_im.save(save_e_path)

    plt.subplot(1, ns, 1 + 1)
    plt.imshow(E_rgb)

    plt.show()

    # Savemat
    mat_dir = os.path.join(save_path, 'm_h_epoch100.mat')
    scipy.io.savemat(mat_dir, {'m_h_result': m_h})

    mat_dir = os.path.join(save_path, 'm_e_epoch100.mat')
    scipy.io.savemat(mat_dir, {'m_e_result': m_e})

    # save variance of M

    mat_dir = os.path.join(save_path, 'sigma_h_epoch100.mat')
    scipy.io.savemat(mat_dir, {'sigma_h_result': sigma_h})

    mat_dir = os.path.join(save_path, 'sigma_e_epoch100.mat')
    scipy.io.savemat(mat_dir, {'sigma_e_result': sigma_e})
