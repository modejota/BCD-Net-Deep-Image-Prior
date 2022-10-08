import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as udata
import glob
import os
# import cv2
import random
import h5py
import numpy as np
import os
import os.path
import h5py
from scipy.io import loadmat
# import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy.io as sio
import hdf5storage
import random
import argparse
import torch.nn.functional as F
from pathlib import Path
# set up device
USE_GPU = True
dtype = torch.float32 # we will be using float throughout this tutorial

######################## generate train/test dataset

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:4')
else:
    device = torch.device('cpu')
print('using device:', device)

parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--datay_root_path", type=str, default='/data/BasesDeDatos/Alsubaie/Data/RGB_images', help="data path")
parser.add_argument("--datam_root_path", type=str, default='/data/BasesDeDatos/Alsubaie/Data/GroundTruth', help="data path")

parser.add_argument("--block_size", type=int, default=64, help="data patch size")
parser.add_argument("--stride", type=int, default=64, help="data patch stride_32fortrain,128forvalid")
parser.add_argument("--train_data_path1", type=str, default='/work/work_ysw/Deblur_Code/Dataset/Train', help="preprocess_data_path")

opt = parser.parse_args()





def main():
    if not os.path.exists(opt.train_data_path1):
        os.makedirs(opt.train_data_path1)

    process_data(block_size=opt.block_size, stride=opt.stride, mode='train')

def directDeconvolve(Y, M):
    #     I=im2uint(I)
    [m, n, c] = Y.shape
    Y = Y.reshape(-1, c).T  # 3xMN
    #     Y=rgb2od(Icol)
    CT = np.linalg.lstsq(M, Y, rcond=None)[0]
    return CT

def rgb2od(rgb):
    od=(-np.log((rgb+1)/256))
    return od

def normalize(data, max_val, min_val):
    return (data-min_val)/(max_val-min_val)

def _PRWGrayImgTensor(img,block_size,stride):

    endc = img.shape[0]
    row = img.shape[1]
    col = img.shape[2]

    if np.mod(row, block_size) == 0:
            row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)

    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)

    row_new = row - row_pad
    col_new = col - col_pad
    row_block = int(row_new / block_size)
    col_block = int(col_new / block_size)
    max_rol=row_new-block_size
    max_col = col_new - block_size
    blocknum = int((max_rol/stride+1) * (max_col/stride+1) )

    imgorg = img[:,0:row_new, 0:col_new]
    Ipadc = imgorg / 255.0
    # Ipadc = imgorg
    img_x = np.zeros([blocknum ,1, block_size, block_size], dtype=np.float32)
    count = 0
    for xi in range(0,max_rol+1,stride):
        for yj in range(0,max_col+1,stride):
            img_x[count] = Ipadc[:,xi :xi +block_size, yj:yj +  block_size]
            count = count + 1


    return img_x


def Im2Patch(img,block_size,stride,M_GT):
    # endc = img.shape[0]
    row = img.shape[0]
    col = img.shape[1]
    endc = img.shape[2]
    if np.mod(row, block_size) == 0:
        row_pad = 0
    else:
        row_pad = block_size - np.mod(row, block_size)

    if np.mod(col, block_size) == 0:
        col_pad = 0
    else:
        col_pad = block_size - np.mod(col, block_size)

    row_new = row - row_pad
    col_new = col - col_pad
    row_block = int(row_new / block_size)
    col_block = int(col_new / block_size)
    max_rol=row_new-block_size
    max_col = col_new - block_size
    blocknum = int((max_rol/stride+1) * (max_col/stride+1) )


    imgorg = img[0:row_new, 0:col_new,:]

    Ipadc = torch.from_numpy( imgorg)
    Ipadc=Ipadc.unsqueeze(0)
    # Ipadc=Ipadc.permute(1,2,0)

    patches_y = np.zeros([blocknum,block_size, block_size,3], dtype=np.float32)

    patches_c= np.zeros([blocknum,2, block_size, block_size], dtype=np.float32)
    patches_m = np.zeros([blocknum, 3, 2], dtype=np.float32)
    count = 0
    for xi in range(0,max_rol+1,stride):
        for yj in range(0,max_col+1,stride):


            img_y = Ipadc[:,xi :xi +block_size, yj:yj +  block_size,:]

            patches_y[count] = img_y

            img_c= directDeconvolve(img_y[0,:,:,:], M_GT)
            img_c=img_c.reshape(2,block_size,block_size)

            patches_c[count]= img_c
            patches_m[count] = M_GT
            count = count + 1
    patches_y=patches_y

    patches_c = patches_c

    patches_m = patches_m
    return patches_y,patches_c,patches_m

def scan_files(datay_root_path,datam_root_path,):
    OD_y_list = []
    OD_m_list = []
    datay_all_dir = os.listdir(datay_root_path)
    datam_all_dir = os.listdir(datam_root_path)
    for i in range(len(datay_all_dir)): #########
        datay_all_dir[i] = os.path.join(datay_root_path, datay_all_dir[i])
    for i in range(len(datay_all_dir)):################len(data_all_dir)

        y_list=glob.glob( datay_all_dir[i] + '/*/*/*.png')+glob.glob(datay_all_dir[i] + '/*/*/*.bmp')
        y_list = sorted([str(x) for x in y_list])
        OD_y_list += y_list

    for i in range(len(datam_all_dir)): #########
        datam_all_dir[i] = os.path.join(datam_root_path, datam_all_dir[i])
    for i in range(len(datam_all_dir)):################len(data_all_dir)
        m_list = glob.glob(datam_all_dir[i] + '/*/SV.mat')+glob.glob(datam_all_dir[i] + '/*/SV.mat')

        m_list = sorted([str(x) for x in m_list])
        OD_m_list += m_list


    return OD_y_list,OD_m_list

def process_data(block_size, stride, mode):
    if mode == 'train':
        print("\nprocess training set ...\n")
        patch_num = 1
        # filenames_hyper = glob.glob(os.path.join(opt.data_path,  '*.png'))
        #
        # filenames_hyper.sort()
        OD_y_list,OD_m_list=scan_files(opt.datay_root_path,opt.datam_root_path)
        # for k in range(1):  # make small dataset
        for k in range(len(OD_y_list)):
            print([OD_y_list[k]])
            img_y = Image.open(OD_y_list[k]);######RGB-gray

            M_GT = loadmat(OD_m_list[k])
            M_GT = M_GT['Stains']
            M_GT = np.array(M_GT, dtype=np.float32)
            img_y= np.array(img_y, dtype=np.float32)
            Y_GT = rgb2od(img_y.astype('float'))
            Y_GT = normalize(Y_GT, max_val=-np.log((1) / 256), min_val=0.)
            # creat patches
            patches_y,patches_c,patches_m = Im2Patch(Y_GT, block_size, stride,M_GT)



            for j in range(patches_y.shape[0]):
                print("generate training sample #%d" % patch_num)
                sub_y = patches_y[j, :, :, :]
                sub_c = patches_c[j, :, :,:]
                sub_m = patches_m[j, :, :]

                train_data_path_array = [opt.train_data_path1]
                random.shuffle(train_data_path_array)
                train_data_path = os.path.join(train_data_path_array[0], 'train'+str(patch_num)+'.mat')
                hdf5storage.savemat(train_data_path, {'rad': sub_y}, format='7.3')
                hdf5storage.savemat(train_data_path, {'mea1': sub_c}, format='7.3')
                hdf5storage.savemat(train_data_path, {'mea2': sub_m}, format='7.3')
                # sio.savemat(train_data_path, {'rad': sub_hyper})
                # sio.savemat(train_data_path, {'mea': sub_y})
                patch_num += 1

        print("\ntraining set: # samples %d\n" % (patch_num-1))





if __name__ == '__main__':
    main()


