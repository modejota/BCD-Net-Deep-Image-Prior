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
from networks.cnet import get_dnet
from networks.mnet import get_knet
from scipy.io import savemat
from options2 import set_opts
import scipy.io
import numpy as np
import os,sys,glob,math

args = set_opts()

use_gpu = True
block_size=64
C = 3
dep_U=4
os.environ['CUDA_LAUNCH_BLOCKING']='2'
# load the pretrained model
print('Loading the Model')
model_dir='./model_mse_sum_kl_test_al0_9/model_80'
checkpoint = torch.load(model_dir)


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

dnet = get_dnet(args.DNet)
knet = get_knet(args.KNet, kernel_size=3)


if use_gpu:
    # dnet = torch.nn.DataParallel(dnet).cuda()
    # knet = torch.nn.DataParallel(knet).cuda()
    dnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)
    # dnet.load_state_dict(checkpoint)
    knet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)
else:
    load_state_dict_cpu(dnet, checkpoint)

dnet.eval()
knet.eval()


data_path = '/work/Camelyon17/work/DECONVOLUCIONES/Original/'

tumor_patches_ids = []
normal_patches_ids = []
unknown_patches_ids=[]
tumor_patches_ids = tumor_patches_ids + glob.glob(data_path + '/*/annotated/*.jpg')
normal_patches_ids = normal_patches_ids + glob.glob(data_path+ '/*/no_annotated/*.jpg')
unknown_patches_ids = unknown_patches_ids + glob.glob(data_path  + '/*/unknown/*.jpg')
OD_list = tumor_patches_ids + normal_patches_ids+unknown_patches_ids
OD_list = sorted([str(x) for x in OD_list])



ImgNum = len(OD_list)

for img_no in range(ImgNum):
    imgName = OD_list[img_no]

    im = img_as_float(Image.open(imgName))

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
        print('Begin Testing on CPU')
    with torch.autograd.set_grad_enabled(False):
        torch.cuda.synchronize()
        tic = time.perf_counter()
        out_KNet1, out_KNet2 = knet(im_y)
        # M = torch.cat((out_KNet1[:, :, 0, :], out_KNet2[:, :, 0, :]), 1)
        # xi = M.reshape(1, -1)

        # im_c = dnet(im_y, xi.detach())

        im_c = dnet(im_y)

        torch.cuda.synchronize()
        toc = time.perf_counter()
        im_c = im_c.cpu().numpy()

        out_KNet1 = out_KNet1.cpu().numpy()
        out_KNet2 = out_KNet2.cpu().numpy()

    # img_name = os.path.split(imgName)[-1]
    # c_dir = os.path.join(save_path,'%s.mat'%img_name)
    # scipy.io.savemat(c_dir, {'C': im_c[0, :2, :row, :col]},appendmat=True)

    re_y = get_od_mearsurement(im_c[:, :2, :row, :col], out_KNet1, out_KNet2, row, col)
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

    # print(f"m_h={out_KNet1[:, :, 0, :]}")
    # print(f"m_e={out_KNet2[:, :, 0, :]}")
    m_out = out_KNet1[:, 0:, :]
    # m_out = np.concatenate((out_KNet1[:, :, 0, :], out_KNet2[:, :, 0, :]), axis=1)
    # #
    display_M(m_out.T)

    # plt.show()

    m_h = out_KNet1[:, :, 0, :].squeeze(0)
    m_e = out_KNet1[:, :, 1, :].squeeze(0)

    sigma_h = out_KNet2[:, 0]
    sigma_e = out_KNet2[:, 1]

    # m_h = out_KNet1.squeeze(0)
    # m_h = m_h.squeeze(0)
    # m_e = out_KNet2.squeeze(0)
    # m_e = m_e.squeeze(0)

    M_read = np.zeros([2, 3])
    M_read[0, :] = m_h[0, :]
    M_read[1, :] = m_e[0, :]
    M_net = M_read.T

    # Savemat
    img = os.path.split(imgName)[-1]

    n = imgName.rfind("center_0")
    img_name=imgName[n:]
    print(img_name)
    im_dir = os.path.dirname(img_name)
    print(dir)

    m_dir = os.path.join('/work/work_ysw/Deblur_Code/test_result/BCDnet_al0.9',im_dir)
    os.makedirs(m_dir, exist_ok=True)
    dir = os.path.join(m_dir, '%s.mat' % img)

    scipy.io.savemat(dir, {'C': im_c[0, :2, :row, :col],'M': M_net,'V':out_KNet2},appendmat=True)
