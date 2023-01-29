import torch
import glob
import cv2
import random
import os
from tqdm import tqdm

import numpy as np

from scipy.io import loadmat

from utils.utils_imgs import npimg_random_crop_patch

class CamelyonDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, centers, patch_size=224, n_samples=None):
        super().__init__()
        self.data_path = data_path
        self.centers = centers
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.image_files = self.scan_files()
        self.img_list = self.load_data()
        self.len = len(self.image_files)
        self.mR = torch.tensor([
                    [0.6442, 0.0928],
                    [0.7166, 0.9541],
                    [0.2668, 0.2831]
                    ])

    def scan_files(self):
        
        tumor_patches_dict = {c : [] for c in self.centers}
        normal_patches_dict = {c : [] for c in self.centers}

        for c in self.centers:
            tumor_patches_dict[c] += glob.glob(self.data_path + 'center_' + str(c) + '/*/annotated/*.jpg')
            normal_patches_dict[c] += glob.glob(self.data_path + 'center_' + str(c) + '/*/no_annotated/*.jpg')

        num_available_patches = np.sum([len(tumor_patches_dict[c]) + len(normal_patches_dict[c]) for c in self.centers])
        print('Available patches:', num_available_patches)
        
        patches_ids = []
        if self.n_samples is not None:
            print(f"Sampling {self.n_samples} patches")
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
            
        return patches_ids
    
    def load_data(self):
        img_list = []
        #od_img_list = []
        pbar = tqdm(self.image_files, total=len(self.image_files))
        pbar.set_description("Loading data")
        for file in pbar:
            img = cv2.imread(file)
            img = img[:,:,::-1] # Changes BGR to RGB

            #if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
            #    img = npimg_random_crop_patch(img, self.patch_size)
            img_list.append(img)

            #od_img = rgb2od_np(img) #Range [0, 5.54]
            #od_img = normalize_to1(od_img, -np.log(1/256), 0) # Range [0, 1]           
            #od_img_list.append(od_img)
        pbar.close()

        return img_list
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.img_list[idx]

        if self.patch_size is not None:
            if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
                img = npimg_random_crop_patch(img, self.patch_size)

        #od_img = rgb2od_np(img) #Range [0, 5.54]
        #od_img = normalize_to1(od_img, -np.log(1/256), 0)
        #od_img = TVTF.to_tensor(od_img.copy()).type(torch.float32)
        #od_img = torch.from_numpy(od_img.copy().transpose(2,0,1).astype(np.float32))
        
        img = torch.from_numpy(img.copy().transpose(2,0,1).astype(np.float32))
        return img, self.mR

class WSSBDatasetTest(torch.utils.data.Dataset):

    def __init__(self, data_path, organ_list=['Lung', 'Breast', 'Colon']):
        self.data_path = data_path
        self.patch_size = np.Inf
        self.organ_list = organ_list
        self.mR = torch.tensor([
                    [0.6442, 0.0928],
                    [0.7166, 0.9541],
                    [0.2668, 0.2831]
                    ])
        self.image_files, self.sv_files = self.scan_files()
        self.img_list, self.M_gt_list = self.load_data()
        self.len = len(self.image_files)        

    def scan_files(self):
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
        return patches_ids, sv_ids

    def load_data(self):
        
        img_list = []
        #od_img_list = []
        pbar = tqdm(self.image_files, total=len(self.image_files))
        pbar.set_description("Loading data")
        for file in pbar:
            img = cv2.imread(file)
            img = img[:,:,::-1].astype("float") # Changes BGR to RGB

            if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
                img = npimg_random_crop_patch(img, self.patch_size)
            img_list.append(img)

            #od_img = rgb2od_np(img) #Range [0, 5.54]
            #od_img = normalize_to1(od_img, -np.log(1/256), 0) # Range [0, 1]           
            #od_img_list.append(od_img)
        pbar.close()

        #C_gt_list = []
        M_gt_list = []
        for i in range(len(self.sv_files)):
            M_gt = loadmat(self.sv_files[i])['Stains']
            #img_od = rgb2od_np(img_list[i])
            #C_gt = direct_deconvolution_np(img_od, M_gt)
            #C_gt_list.append(C_gt)
            M_gt_list.append(M_gt)
        return img_list, M_gt_list
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.img_list[idx]
        #od_img = self.od_img_list[idx]
        mR = self.mR
        img = torch.from_numpy(img.copy().transpose(2,0,1).astype(np.float32))
        #od_img = TVTF.to_tensor(od_img.copy()).type(torch.float32)
        #C_gt = torch.from_numpy(self.C_gt_list[idx]).type(torch.float32)
        M_gt = torch.from_numpy(self.M_gt_list[idx]).type(torch.float32)
        #C_gt_rgb = torch.from_numpy(self.C_gt_rgb_list[idx]).type(torch.float32)
        return img, mR, M_gt

class FilesDataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, patch_size=None, n_samples=None):
        super().__init__()
        self.data_path = data_path
        self.patch_size = patch_size
        self.n_samples = n_samples
        print("Scanning files...")
        self.image_files = self.scan_files()
        print("Loading data...")
        self.img_list = self.load_data()
        self.len = len(self.image_files)

    def scan_files(self):

        files_vec = []
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg', 'tif', '.tiff', '.bmp', '.gif')): 
                    files_vec.append(os.path.join(root, file))
        files_vec = list(set(files_vec))
        print("Found {} files".format(len(files_vec)))
        if self.n_samples is not None:
            files_vec = files_vec[:self.n_samples]
        return files_vec
    
    def load_data(self):
        img_list = []
        #od_img_list = []
        for file in self.image_files:
            img = cv2.imread(file)
            img = img[:,:,::-1] # Changes BGR to RGB

            img_list.append(img)

        return img_list
    
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img = self.img_list[idx]
        file = self.image_files[idx]

        if self.patch_size is not None:
            if (self.patch_size < img.shape[0]) or (self.patch_size < img.shape[1]):
                img = npimg_random_crop_patch(img, self.patch_size)

        #od_img = rgb2od_np(img) #Range [0, 5.54]
        #od_img = normalize_to1(od_img, -np.log(1/256), 0)
        
        #od_img = torch.from_numpy(od_img.copy().transpose(2,0,1).astype(np.float32))
        img = torch.from_numpy(img.copy().transpose(2,0,1).astype(np.float32))
        return img, file


