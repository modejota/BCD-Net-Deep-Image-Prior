import cv2
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVTF
from pathlib import Path
import random
from utils.utils_imgs import npimg_random_augmentation, tensorimg_add_noise,tensorimg_random_crop_patch, npimg_random_crop_pair_patch
import glob
import csv
def im2uint(I):
    I2=I*255
    return I2.astype('int')
# def rgb2od(rgb):
#     od = (-torch.log((rgb + 1) / 256))
#     return od
def rgb2od(rgb):
    od=(-np.log((rgb+1)/256))
    return od
def normalize(data, max_val, min_val):
    return (data - min_val) / (max_val - min_val)
class DeconvolutionTrainDataset(Dataset):
    def __init__(self, data_path, patch_size, augment=False, add_noise=False):
        super().__init__()
        data_root_path = data_path
        self.patch_size = patch_size
        self.augment = augment
        self.add_noise = add_noise
        self.OD_img_list= self.scan_files(data_root_path)

    def scan_files(self, camelyon_dir):
        OD_img_list = []
        n_samples = 60000  # Desired size for the dataset, you can change this
        #         camelyon_dir='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/'
        train_centers = [0, 1,2,3, 4]  # This will take images from centers 0, 2 and 4
        tumor_patches_ids = []
        normal_patches_ids = []
        unknown_patches_ids = []
        for c in train_centers:
            tumor_patches_ids = tumor_patches_ids + glob.glob(
                camelyon_dir + 'center_' + str(c) + '/*/annotated/*.jpg')
            normal_patches_ids = normal_patches_ids + glob.glob(
                camelyon_dir + 'center_' + str(c) + '/*/no_annotated/*.jpg')
            unknown_patches_ids = unknown_patches_ids + glob.glob(
                camelyon_dir + 'center_' + str(c) + '/*/unknown/*.jpg')
        # ALL PATCHES
        train_patches = tumor_patches_ids + normal_patches_ids + unknown_patches_ids
        # print(len(train_patches))
        random.seed(42)  # This is important to choose always the same patches
        OD_img_list = random.sample(train_patches, n_samples)
        len(OD_img_list)
        with open(r"OD_img_list.csv", 'w+', newline='') as file:

            writer = csv.writer(file)
            writer.writerows(OD_img_list)
        return OD_img_list

    def __len__(self):
        return len(self.OD_img_list)



    def __getitem__(self, index):
        OD_img = Image.open(self.OD_img_list[index]);#####RGB
        # OD_img = cv2.imread(self.OD_img_list[index])######BGR
        OD_img = np.array(OD_img, dtype=np.uint8)

        # M_data = torch.tensor([[[0.39579749, 0.44028017, 0.16392234],[0.06977444, 0.71736842, 0.21285714]]])#####before 3.7

        M_data = torch.tensor([[[0.6442,0.7166,0.2668], [0.0928,0.9541,0.2831]]])

        OD_img = tensorimg_random_crop_patch(OD_img, self.patch_size)

        if self.augment:
            OD_img= npimg_random_augmentation(OD_img)

        OD_img = OD_img.astype('float')#########
        OD_img = rgb2od(OD_img)
        OD_img=np.transpose(OD_img, (2,0,1))

        # OD_img = normalize(OD_img, max_val=np.max(OD_img), min_val=np.min(OD_img))



        OD_img = torch.from_numpy(OD_img.copy())
        OD_img = OD_img.type(torch.FloatTensor)

        # OD_img = normalize(OD_img, max_val=-np.log((1)/256), min_val=0.)
        # OD_img  = OD_img/255

        if self.add_noise:
            r = random.randint(0, 2)
            if r == 0:
                OD_img = tensorimg_add_noise(OD_img, sigma=1.0, rgb_range=255)
            else:
                OD_img = OD_img
        return OD_img, M_data


class DeconvolutionTestDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        n_samples = 6000  # Desired size for the dataset, you can change this
        #         camelyon_dir='/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/'
        train_centers = [1,3]  # This will take images from centers 0, 2 and 4
        tumor_patches_ids = []
        normal_patches_ids = []
        unknown_patches_ids = []
        for c in train_centers:
            tumor_patches_ids = tumor_patches_ids + glob.glob(
                data_path + 'center_' + str(c) + '/*/annotated/*.jpg')
            normal_patches_ids = normal_patches_ids + glob.glob(
                data_path + 'center_' + str(c) + '/*/no_annotated/*.jpg')
            unknown_patches_ids = unknown_patches_ids + glob.glob(
                data_path + 'center_' + str(c) + '/*/unknown/*.jpg')
        # ALL PATCHES
        train_patches = tumor_patches_ids + normal_patches_ids + unknown_patches_ids
        # print(len(train_patches))
        random.seed(42)  # This is important to choose always the same patches
        self.OD_list = random.sample(train_patches, n_samples)
        # OD_data_path = Path(data_path+'patient_099_node_4/annotated')
        # OD_list = list(OD_data_path.glob('*.jpg'))
        # self.OD_list = sorted([str(x) for x in OD_list])

    def __len__(self):
        return len(self.OD_list)

    def __getitem__(self, index):
        OD_file_name = self.OD_list[index]
        OD_img = Image.open(self.OD_list[index]);  #####RGB
        # OD_img = cv2.imread(self.OD_img_list[index])######BGR
        OD_img = np.array(OD_img, dtype=np.uint8)


        OD_img = OD_img.astype('float')  ##
        OD_img = rgb2od(OD_img)
        OD_img = np.transpose(OD_img, (2, 0, 1))
        # OD_img = normalize(OD_img, max_val=np.max(OD_img), min_val=np.min(OD_img))


        OD_img = torch.from_numpy(OD_img.copy())
        OD_img = OD_img.type(torch.FloatTensor)


        # OD_img = normalize(OD_img, max_val=-np.log((1)/256), min_val=0.)
        # OD_img = OD_img / 255
        # M_data = torch.tensor([[[0.39579749, 0.44028017, 0.16392234],[0.06977444, 0.71736842, 0.21285714]]])
        M_data = torch.tensor([[[0.6442, 0.7166, 0.2668], [0.0928, 0.9541, 0.2831]]])
        return OD_img, M_data


if __name__ == "__main__":
    dataset = DeconvolutionTestGoProDataset(data_path = "/data/")
    print(len(dataset.clean_list))
    print(len(dataset.blurry_list))
    print(dataset.clean_list[10:15])
    print(dataset.blurry_list[10:15])