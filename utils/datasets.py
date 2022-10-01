import torch
import glob
import cv2
import random

import numpy as np
import torchvision.transforms.functional as TVTF

from utils.utils_imgs import npimg_random_crop_patch
from utils.utils_BCD import np_rgb2od, normalize_to1

class OD_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, centers, patch_size=224, n_samples=None):
        super().__init__()
        self.data_path = data_path
        self.centers = centers
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.image_files = self.scan_files()
        self.img_list, self.od_img_list = self.load_data()
        self.len = len(self.image_files)
        self.mR = torch.tensor([[
                    [0.6442, 0.0928],
                    [0.7166, 0.9541],
                    [0.2668, 0.2831]
                    ]])

    def scan_files(self):
        return None
    
    def load_data(self):
        img_list = []
        od_img_list = []
        for file in self.image_files:
            img = cv2.imread(file)
            img = img[:,:,::-1] # Changes BGR to RGB

            if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
                img = npimg_random_crop_patch(img, self.patch_size)

            od_img = np_rgb2od(img) #Range [0, 5.54]
            od_img = normalize_to1(od_img, -np.log(1/256), 0) # Range [0, 1]

            img_list.append(img)
            od_img_list.append(od_img)

        return img_list, od_img_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.img_list[idx]
        od_img = self.od_img_list[idx]
        
        od_img = TVTF.to_tensor(od_img.copy()).type(torch.float32)
        img = TVTF.to_tensor(img.copy()).type(torch.float32)

        return img, od_img, self.mR

class CamelyonDataset(OD_Dataset):
       

    def scan_files(self):

        tumor_patches_ids = []
        normal_patches_ids = []
        for center in self.centers:
            tumor_patches_ids = tumor_patches_ids + glob.glob(self.data_path + 'center_' + str(center) + '/*/annotated/*.jpg')
            normal_patches_ids = normal_patches_ids + glob.glob(self.data_path + 'center_' + str(center) + '/*/no_annotated/*.jpg')

        # ALL PATCHES
        patches_ids = tumor_patches_ids + normal_patches_ids
        print('Available patches:', len(patches_ids))
        if self.n_samples is not None:
            if self.n_samples < len(patches_ids):
                random.seed(42)  # This is important to choose always the same patches
                patches_ids = random.sample(patches_ids, self.n_samples)
        return patches_ids

class WSSBDatasetTest(torch.utils.data.Dataset):

    def __init__(self, data_path, patch_size=224, n_samples=None, organ_list=['Lung', 'Breast', 'Colon']):
        super().__init__(data_path, [], patch_size, n_samples)
        self.organ_list = organ_list

    def scan_files(self):
        patches_ids = []
        for organ in self.organ_list:
            patches_ids = patches_ids + glob.glob(self.data_path + organ + '/*/*/*.png')
        return patches_ids