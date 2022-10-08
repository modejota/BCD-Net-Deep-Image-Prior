import cv2
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVTF
from pathlib import Path
import random
from utils.utils_imgs import npimg_random_augmentation, tensorimg_add_noise, npimg_random_crop_pair_patch


class DeblurTrainGoProDataset(Dataset):
    def __init__(self, data_path, patch_size, augment=True, add_noise=False):
        super().__init__()
        data_root_path = data_path + "train"
        self.patch_size = patch_size
        self.augment = augment
        self.add_noise = add_noise
        self.blurry_list, self.clean_list, self.kernel_list = self.scan_files(data_root_path)

    def scan_files(self, data_root_path):
        blurry_list = []
        clean_list = []
        kernel_list = []
        data_all_dir = os.listdir(data_root_path)
        for i in range(len(data_all_dir)):
            data_all_dir[i] = os.path.join(data_root_path, data_all_dir[i])
        for i in range(len(data_all_dir)):
            blurry_path = Path(data_all_dir[i] + '/blur')
            clean_path = Path(data_all_dir[i] + '/sharp')
            b_list = list(blurry_path.glob('*.png'))
            b_list = sorted([str(x) for x in b_list])
            blurry_list += b_list
            c_list = list(clean_path.glob('*.png'))
            c_list = sorted([str(x) for x in c_list])
            clean_list += c_list
            k_list = list(blurry_path.glob('*kernel.png'))
            k_list = sorted([str(x) for x in k_list])
            kernel_list += k_list
        blurry_list_s = [b for b in blurry_list if b not in kernel_list]
        kernel_list =  [b[:-4] + "_kernel.png" for b in blurry_list_s]
        return blurry_list_s, clean_list, kernel_list

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, index):
        clean_img = cv2.imread(self.clean_list[index])[:, :, ::-1]
        blurry_img = cv2.imread(self.blurry_list[index])[:, :, ::-1]
        kernel = cv2.imread(self.kernel_list[index])[:, :, ::-1]

        clean_img, blurry_img = npimg_random_crop_pair_patch(clean_img, blurry_img, self.patch_size)
        if self.augment:
            clean_img, blurry_img, kernel = npimg_random_augmentation(clean_img, blurry_img, kernel)

        clean_img = TVTF.to_tensor(clean_img.copy())
        blurry_img = TVTF.to_tensor(blurry_img.copy())
        kernel = TVTF.to_tensor(kernel.copy())[0]  # 3 x k x k ->  k x k
        kernel = torch.div(kernel, torch.sum(kernel))  # sum to 1
        if self.add_noise:
            r = random.randint(0, 2)
            if r == 0:
                blurry_img = tensorimg_add_noise(blurry_img, sigma=1.0, rgb_range=255)
            else:
                blurry_img = blurry_img
        return clean_img, blurry_img, kernel


class DeblurTestGoProDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        clean_data_path = Path(data_path + '/test/gt/')
        clean_list = list(clean_data_path.glob('*.png'))
        self.clean_list = sorted([str(x) for x in clean_list])
        blur_data_path = Path(data_path + '/test/blur/')
        blur_list = list(blur_data_path.glob('*.png'))
        self.blurry_list = sorted([str(x) for x in blur_list])

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, index):
        blurry_file_name = self.blurry_list[index]
        clean_img = cv2.imread(self.clean_list[index])[:, :, ::-1]
        blurry_img = cv2.imread(self.blurry_list[index])[:, :, ::-1]

        clean_img = TVTF.to_tensor(clean_img.copy())
        blurry_img = TVTF.to_tensor(blurry_img.copy())
        kernel = torch.zeros(31, 31)

        return clean_img, blurry_img, kernel, blurry_file_name


if __name__ == "__main__":
    dataset = DeblurTestGoProDataset(data_path = "../data/gopro_patch_kernels")
    print(len(dataset.clean_list))
    print(len(dataset.blurry_list))
    print(dataset.clean_list[10:15])
    print(dataset.blurry_list[10:15])