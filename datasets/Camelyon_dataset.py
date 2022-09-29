import cv2
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVTF
from pathlib import Path
import random
import glob
from utils.utils_imgs import npimg_random_augmentation, tensorimg_add_noise, tensorimg_random_crop_patch, npimg_random_crop_patch
from utils.utils_BCD import np_rgb2od, normalize_to1


class CamelyonDataset(Dataset):
    def __init__(self, data_path, train_centers, patch_size, n_samples=None):
        super().__init__()
        data_root_path = data_path
        self.train_centers=train_centers
        self.patch_size = patch_size
        self.n_samples = n_samples

        self.image_list = self.scan_files(data_root_path, train_centers)
        self.mR = torch.tensor([[[0.6442, 0.0928],
                                 [0.7166, 0.9541],
                                 [0.2668, 0.2831]]])
        

    def scan_files(self, data_root_path, train_centers):
        # /center_*/patient_*_node_*/annotated/*.jpg
        # image_list=glob.glob(data_root_path + '/center_0/*/*/*.jpg')
        # print('patch list len:',len(image_list))
        # return image_list
        OD_img_list = []

        tumor_patches_ids = []
        normal_patches_ids = []
        for c in train_centers:
            tumor_patches_ids = tumor_patches_ids + glob.glob(data_root_path + 'center_' + str(c) + '/*/annotated/*.jpg')
            normal_patches_ids = normal_patches_ids + glob.glob(data_root_path + 'center_' + str(c) + '/*/no_annotated/*.jpg')

        # ALL PATCHES
        train_patches = tumor_patches_ids + normal_patches_ids
        print('Available patches:', len(train_patches))
        if self.n_samples is not None:
            if self.n_samples < len(train_patches):
                OD_img_list = random.sample(train_patches, self.n_samples)
        else:
            random.seed(42)  # This is important to choose always the same patches
            OD_img_list = train_patches
        return OD_img_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        observed_image = cv2.imread(self.image_list[index]) #Warning BGR mode!
        observed_image = observed_image[:,:,::-1] #Changes BGR to RGB
        observed_image = npimg_random_crop_patch(observed_image, self.patch_size)

        od_img = np_rgb2od(observed_image) #Range [0, 5.54]
        od_img = normalize_to1(od_img, -np.log(1/256), 0) # Range [0,1]
        #od_img = od_img.astype('float')
        od_img = TVTF.to_tensor(od_img.copy())
        od_img = od_img.type(torch.float32)

        return od_img, self.mR

