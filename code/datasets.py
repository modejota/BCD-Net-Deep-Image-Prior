import torch
import glob
import cv2
import random
import os

import numpy as np

from scipy.io import loadmat

def random_crop(self, img, patch_size):
        h, w, _ = img.shape
        if h > patch_size:
            h1 = random.randint(0, h-patch_size)
        else:
            h1 = 0
        if w > patch_size:
            w1 = random.randint(0, w-patch_size)
        else:
            w1 = 0
        return img[h1:h1+patch_size, w1:w1+patch_size, :]

class CamelyonDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, centers, patch_size=224, n_samples=None):
        super().__init__()
        self.data_path = data_path
        self.centers = centers
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.image_files = self.scan_files()

    def scan_files(self):

        print('[CamelyonDataset] Scanning files...')
        
        tumor_patches_dict = {c : [] for c in self.centers}
        normal_patches_dict = {c : [] for c in self.centers}

        for c in self.centers:
            tumor_patches_dict[c] += glob.glob(self.data_path + 'center_' + str(c) + '/*/annotated/*.jpg')
            normal_patches_dict[c] += glob.glob(self.data_path + 'center_' + str(c) + '/*/no_annotated/*.jpg')

        num_available_patches = np.sum([len(tumor_patches_dict[c]) + len(normal_patches_dict[c]) for c in self.centers])
        print('[CamelyonDataset] Available patches:', num_available_patches)
        
        patches_ids = []
        if self.n_samples is not None:
            print(f"[CamelyonDataset] Sampling {self.n_samples} patches")
            if self.n_samples < num_available_patches:
                random.seed(42)
                n_samples_per_center = self.n_samples // len(self.centers)
                for c in self.centers:
                    patches_ids += random.sample(tumor_patches_dict[c], n_samples_per_center//2)
                    patches_ids += random.sample(normal_patches_dict[c], n_samples_per_center//2)
            else:
                for c in self.centers:
                    patches_ids += tumor_patches_dict[c]
                    patches_ids += normal_patches_dict[c]
        else:
            for c in self.centers:
                patches_ids += tumor_patches_dict[c]
                patches_ids += normal_patches_dict[c]
        
        print('[CamelyonDataset] Done scanning files')

        return patches_ids
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file = self.image_files[idx]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.patch_size is not None:
            if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
                img = random_crop(img, self.patch_size)
        
        img = torch.from_numpy(img.copy().transpose(2,0,1).astype(np.float32))
        return img

class WSSBDatasetTest(torch.utils.data.Dataset):

    def __init__(self, data_path, organ_list=['Lung', 'Breast', 'Colon']):
        self.data_path = data_path
        self.patch_size = np.Inf
        self.organ_list = organ_list
        self.image_files, self.sv_files = self.scan_files()

    def scan_files(self):
        print('[WSSBDataset] Scanning files...')
        patches_ids = []
        sv_ids = []
        for organ in self.organ_list:
            c_dir_list = os.listdir(f"{self.data_path}/GroundTruth/{organ}/")
            for c_dir in c_dir_list:
                id_dir_list = os.listdir(f"{self.data_path}/RGB_images/{organ}/{c_dir}/")
                for id_dir in id_dir_list:
                    sv_ids.append(f"{self.data_path}/GroundTruth/{organ}/{c_dir}/SV.mat")
                    img_dir_path = f"{self.data_path}/RGB_images/{organ}/{c_dir}/{id_dir}/"
                    patches_ids = patches_ids + [f"{img_dir_path}/{name}" for name in os.listdir(img_dir_path) if name.endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print('[WSSBDataset] Done scanning files')
        return patches_ids, sv_ids
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_file = self.image_files[idx]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
            img = random_crop(img, self.patch_size)
        img = torch.from_numpy(img.copy().transpose(2,0,1).astype(np.float32))

        sv_file = self.sv_files[idx]
        M_gt = torch.from_numpy(loadmat(sv_file)['Stains'].astype(np.float32))

        return img, M_gt