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
from loss_norm import loss_fn
from networks.dnet import get_dnet
from networks.knet import get_knet
from scipy.io import savemat
from options import set_opts
import scipy.io
import numpy as np
import os,sys,glob,math
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from scipy.io import loadmat
from utils.utils_imgs import WSSBDatasetTest
from torch.utils.data import DataLoader


args = set_opts()

use_gpu = True
block_size=64
C = 3
dep_U=4
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')
# load the pretrained model
print('Loading the Model')
model_dir='./model_mse_kl_test_al0_2/model_60'

dnet = get_dnet(args.DNet)
knet = get_knet(args.KNet, kernel_size=3)
checkpoint = torch.load(model_dir,map_location='cpu')
state_dict = torch.load(model_dir,map_location='cpu')

dnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)

knet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)

def get_od_mearsurement(out_DNet, out_KNet1,out_KNet2,batch_size,patch_size):
    # M = np.concatenate((out_KNet1[:, :, 0, :], out_KNet2[:, :, 0, :]), axis=1)
    M = out_KNet1[:, 0, :, :]
    M=M[0,:,:].T
    C=out_DNet.reshape(2,patch_size*patch_size)

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






if use_gpu:
    # dnet = torch.nn.DataParallel(dnet).cuda()
    # knet = torch.nn.DataParallel(knet).cuda()
    dnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)
    # dnet.load_state_dict(checkpoint)
    knet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)
else:
    dnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)
    knet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)

dnet.eval()
knet.eval()



data_path = '/work/work_ysw/Deblur_Code/test_data/RGB_images/'
save_path='/work/work_ysw/Deblur_Code/test_result/RGB_images_model_withoutalpha/'

#
# train_centers = [1]  # This will take images from centers 0, 2 and 4
# # Breast_patches_ids = []
# # Colon_patches_ids = []
# Lung_patches_ids=[]
# for c in train_centers:
#     # Breast_patches_ids = Breast_patches_ids + glob.glob(
#     #     data_path + 'Breast' +  '/*/*/*.png')
#     # Colon_patches_ids = Colon_patches_ids + glob.glob(
#     #     data_path + 'Colon' +  '/*/*/*.bmp')
#     Lung_patches_ids = Lung_patches_ids + glob.glob(
#         data_path + 'Lung' +  '/*/*/*.png')
#
#
#
#         # ALL PATCHES
# OD_list = Lung_patches_ids
# OD_list = sorted([str(x) for x in OD_list])
#
# # im_path = '/work/work_ysw/Deblur_Code/test_result/BCDnet/center_1'
# #
# # filepaths = glob.glob(im_path + '/*/annotated/*.jpg')
# #
# # ImgNum = len(filepaths)
# ImgNum=len(OD_list)
test_set = WSSBDatasetTest(mode='test')  ########### load test image
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)


test_number = 0
mean_psrn = 0
mean_ssim = 0

# tic = time.perf_counter()
for t, im in enumerate(test_loader):
    im = im.squeeze(dim=0)
    # H, W, _ = im.shape
    # if H % 2 ** dep_U != 0:
    #     H -= H % 2 ** dep_U
    # if W % 2 ** dep_U != 0:
    #     W -= W % 2 ** dep_U
    #
    # row = im.shape[0]
    # col = im.shape[1]
    #
    # if np.mod(row, block_size) == 0:
    #     row_pad = 0
    # else:
    #     row_pad = block_size - np.mod(row, block_size)
    #
    # if np.mod(col, block_size) == 0:
    #     col_pad = 0
    # else:
    #     col_pad = block_size - np.mod(col, block_size)
    # row_new = row + row_pad
    # col_new = col + col_pad
    #
    # max_rol = row_new - block_size
    # max_col = col_new - block_size
    #
    # im_y1 = im[row - row_pad:, col - col_pad:, :]
    # im_y = np.zeros((row_new, col_new, 3), float)
    # im_y[:row, :col, :] = im
    # im_y[row:row_new, col:col_new, :] = im_y1
    #
    # im_y = im[:H, :W, ]

    im_y = rgb2od(im )


    # im_y = normalize(im_y, max_val=np.max(im_y), min_val=np.min(im_y))
    if use_gpu:
        im_y = torch.tensor(im_y).to(device)
        im_y = torch.permute(im_y, [0,3,1,2])
        # im_y = im_y.unsqueeze(0)
        im_y = im_y.type(torch.FloatTensor)
        print('Begin Testing on GPU')
    else:
        im_y = torch.from_numpy(im_y)
        im_y = torch.permute(im_y, [2, 0, 1])
        im_y = im_y.unsqueeze(0)
        im_y = im_y.type(torch.FloatTensor)
        print('Begin Testing on CPU')
    with torch.autograd.set_grad_enabled(False):

        tic = time.perf_counter()
        out_KNet1, out_KNet2 = knet(im_y)

        im_c = dnet(im_y)

        toc = time.perf_counter()
        im_c = im_c.cpu().numpy()

        out_KNet1 = out_KNet1.cpu().numpy()
        out_KNet2 = out_KNet2.cpu().numpy()

    # img_name = os.path.split(imgName)[-1]
    # c_dir = os.path.join(save_path,'%s.mat'%img_name)
    # scipy.io.savemat(c_dir, {'C': im_c[0, :2, :row, :col]},appendmat=True)


    # img_name = os.path.split(imgName)[-4:-1]

#     dir = os.path.dirname(imgName)
#     print(dir)
    # m_dir = os.path.join(dir, '%s.mat' % img_name)
    # scipy.io.savemat(m_dir, {'C': im_c[0, :2, :row, :col], 'M': M_net}, appendmat=True)

    # mat_dir = os.path.join(save_path, dir,'c_epoch100.mat')
    # scipy.io.savemat('c_epoch100.mat', {'c_result': im_c[0, :2, :row, :col]})
    #
    # re_y = get_od_mearsurement(im_c[:, :2, :row, :col], out_KNet1, out_KNet2, row, col)
    # re_y = od2rgb(re_y)
    # re_y = re_y / 255
    #
    # plt.subplot(231)
    # plt.imshow(im)
    # plt.title('Y Image')
    #
    # plt.subplot(232)
    # plt.imshow(im_c[0, 0, :row, :col])
    # plt.title('H Image')
    #
    # plt.subplot(233)
    # plt.imshow(im_c[0, 1, :row, :col])
    # plt.title('E Image')
    #
    # plt.subplot(234)
    # plt.imshow(re_y)
    # plt.title('Y_re Image')
    # plt.show()
# toc = time.perf_counter()
    # print(f"m_h={out_KNet1[:, :, 0, :]}")
    # print(f"m_e={out_KNet1[:, :, 1, :]}")
print('time={:.4f}'.format(toc - tic))
    #
    # # m_out=np.concatenate((out_KNet1[:, :,0,:],out_KNet2[:, :,0,:]),axis=1)
    # m_out = out_KNet1[:, 0:, :]
    # # #
    # display_M(m_out.T)
    #
    # plt.show()
    #
    # m_h = out_KNet1[:, :, 0, :].squeeze(0)
    # # m_h=m_h.squeeze(0)
    # m_e = out_KNet1[:, :, 1, :].squeeze(0)
    # # m_e=m_e.squeeze(0)
    #
    # M_read = np.zeros([2, 3])
    # M_read[0, :] = m_h[0, :]
    # M_read[1, :] = m_e[0, :]
    # M_net = M_read.T
    #
    # C_norm = im_c.reshape(2, -1)
    #
    # C_net = C_norm
    # [ns, w] = C_net.shape
    # new_w = int(w ** 0.5)
    #
    # save_h_path = os.path.join(save_path,dir,'H.png')
    #
    # H_od = np.dot(M_net[:, 0].reshape(3, 1), C_net[0, :].reshape(1, -1))
    # H_rgb = od2rgb(H_od).astype('int')
    # H_rgb = H_rgb.reshape(3, new_w, new_w).T
    # H_im = np.clip(H_rgb, 0, 255)
    # H_im = Image.fromarray(H_im.astype(np.uint8), mode='RGB')
    # H_im.save(save_h_path)
    #
    # plt.subplot(1, ns, 0 + 1)
    # plt.imshow(H_rgb)
    #
    # save_e_path = os.path.join(save_path,dir, 'E.png')
    # E_od = np.dot(M_net[:, 1].reshape(3, 1), C_net[1, :].reshape(1, -1))
    # E_rgb = od2rgb(E_od).astype('int')
    # E_rgb = E_rgb.reshape(3, new_w, new_w).T
    # E_im = np.clip(E_rgb, 0, 255)
    # E_im = Image.fromarray(E_im.astype(np.uint8), mode='RGB')
    # E_im.save(save_e_path)
    #
    # plt.subplot(1, ns, 1 + 1)
    # plt.imshow(E_rgb)
    #
    # plt.show()
    #
    # # Savemat
    # mat_dir = os.path.join(save_path, 'm_h_epoch100.mat')
    # scipy.io.savemat(mat_dir, {'m_h_result': m_h})
    #
    # mat_dir = os.path.join(save_path, 'm_e_epoch100.mat')
    # scipy.io.savemat(mat_dir, {'m_e_result': m_e})
    #
    # # #Savemat
    # # mat_dir = os.path.join(save_path, 'm_epoch100.mat')
    # # scipy.io.savemat(mat_dir, {'m_result': m_out})
    #
    # gt_folder = '/data/BasesDeDatos/Alsubaie/Data/GroundTruth/Lung/2/'
    # M_GT = loadmat(gt_folder + 'SV.mat')
    # M_GT = M_GT['Stains']
    #
    # # patch = cv2.imread(im_path)
    #
    # # CV2 lee en BGR en lugar de RGB
    # # I_GT = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    #
    # Y_GT = rgb2od(im.astype('float'))
    # get_range(Y_GT)
    # norm_CONST = -np.log(1 / 256)
    #
    # Y_GT_norm = Y_GT / norm_CONST
    #
    # norm_CONST = -np.log(1 / 256)
    # C_GT_norm = directDeconvolve(Y_GT_norm.astype('float'), M_GT)
    # C_GT = C_GT_norm * norm_CONST
    # stains_GT = get_C_rgb(C_GT, M_GT)
    #
    # print('NETWORK')
    # all_images = get_C_rgb(C_net, M_net)
    # psnr_H = peak_signal_noise_ratio(stains_GT[0, :, :, :], all_images[0, :, :, :])
    # psnr_E = peak_signal_noise_ratio(stains_GT[1, :, :, :], all_images[1, :, :, :])
    # print('PSNR', psnr_H, psnr_E)
    #
    # ssim_H = structural_similarity(stains_GT[0, :, :, :], all_images[0, :, :, :], multichannel=True)
    # ssim_E = structural_similarity(stains_GT[1, :, :, :], all_images[1, :, :, :], multichannel=True)
    #
    # print('SSIM', ssim_H, ssim_E)
