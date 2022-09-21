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
from utils.utils_imgs import npimg_random_augmentation, tensorimg_add_noise,tensorimg_random_crop_patch, npimg_random_crop_pair_patch
from utils.utils_BCD import np_rgb2od, normalize_to1


class CamelyonDataset(Dataset):
    def __init__(self, data_path, train_centers, patch_size, augment=False, add_noise=False):
        super().__init__()
        data_root_path = data_path
        self.train_centers=train_centers
        self.patch_size = patch_size
        self.augment = augment
        self.add_noise = add_noise
        self.image_list = self.scan_files(data_root_path,train_centers)
        self.mR = torch.tensor([[0.6442, 0.0928],
             [0.7166, 0.9541],
             [0.2668, 0.2831]])
        # print('Reference matrix',self.mR,self.mR.size())

    def scan_files(self, data_root_path, train_centers):
        # /center_*/patient_*_node_*/annotated/*.jpg
        # image_list=glob.glob(data_root_path + '/center_0/*/*/*.jpg')
        # print('patch list len:',len(image_list))
        # return image_list
        OD_img_list = []
        n_samples = 4  # Desired size for the dataset, you can change this
        #         camelyon_dir='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/'
        # train_centers = [0, 2, 4]  # This will take images from centers 0, 2 and 4
        tumor_patches_ids = []
        normal_patches_ids = []
        for c in train_centers:
            tumor_patches_ids = tumor_patches_ids + glob.glob(data_root_path + 'center_' + str(c) + '/*/annotated/*.jpg')
            normal_patches_ids = normal_patches_ids + glob.glob(
                data_root_path + 'center_' + str(c) + '/*/no_annotated/*.jpg')

        # ALL PATCHES
        train_patches = tumor_patches_ids + normal_patches_ids
        print('Available patches:', len(train_patches))
        random.seed(42)  # This is important to choose always the same patches

        OD_img_list = random.sample(train_patches, n_samples)
        # len(OD_img_list)
        return OD_img_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        observed_image = cv2.imread(self.image_list[index]) #Warning BGR mode!
        observed_image=observed_image[:,:,::-1] #Changes BGR to RGB
        observed_image = tensorimg_random_crop_patch(observed_image, self.patch_size)
        od_img= np_rgb2od(observed_image) #Range [0, 5.54]
        od_img= normalize_to1(od_img,-np.log(1/256),0) #Range [0,1]
        od_img=od_img.astype('float')
        od_img = TVTF.to_tensor(od_img)
        od_img= od_img.type(torch.FloatTensor)

        return od_img, self.mR

