import cv2
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVTF
from pathlib import Path
from utils.utils_imgs import npimg_random_augmentation, npimg_random_crop_pair_patch


class DeblurTrainRealBlurDataset(Dataset):
    def __init__(self, data_path, patch_size, mode="J", augment=True, add_BSD=False):
        super().__init__()
        data_root_dir = data_path + "RealBlur_" + mode + "/train"
        self.patch_size = patch_size
        self.augment = augment
        self.blurry_list, self.clean_list, self.kernel_list = self.scan_files(data_root_dir)

        BSD_blurry_list, BSD_clean_list, BSD_kernel_list = self.scan_BSD_flies()

        if add_BSD:
            self.blurry_list = self.blurry_list + BSD_blurry_list
            self.clean_list = self.clean_list + BSD_clean_list
            self.kernel_list = self.kernel_list + BSD_kernel_list

        assert len(self.clean_list) == len(self.kernel_list) and len(self.kernel_list) == len(
            self.blurry_list), "Error!"

    def scan_BSD_flies(self):
        blur_dir = Path("../data/BSD/blur")
        clean_dir = Path("../data/BSD/gt")

        b_list = list(blur_dir.glob('*blurred.png'))
        b_list = sorted([str(x) for x in b_list])

        c_list = list(clean_dir.glob('*.png'))
        c_list = sorted([str(x) for x in c_list])

        k_list = list(blur_dir.glob('*predkernel.png'))
        k_list = sorted([str(x) for x in k_list])

        return b_list, c_list, k_list

    def scan_files(self, data_root_path):
        blur_dir = Path(data_root_path + "/blur")
        clean_dir = Path(data_root_path + "/gt")

        b_list = list(blur_dir.glob('*.png'))
        b_list = sorted([str(x) for x in b_list])

        c_list = list(clean_dir.glob('*.png'))
        c_list = sorted([str(x) for x in c_list])

        k_list = list(blur_dir.glob('*kernel.png'))
        k_list = sorted([str(x) for x in k_list])
        blurry_list_s = [b for b in b_list if b not in k_list]
        kernel_list =  [b[:-4] + "_kernel.png" for b in blurry_list_s]
        return blurry_list_s, c_list, kernel_list

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
        kernel = TVTF.to_tensor(kernel.copy())[0]  # 3 x k x k -> k x k
        kernel = torch.div(kernel, torch.sum(kernel)) # sum to 1

        return clean_img, blurry_img, kernel


class DeblurTestRealBlurDataset(Dataset):
    def __init__(self, data_path, mode="J"):
        super().__init__()
        data_root_dir = data_path + "RealBlur_" + mode + "/test"
        self.blurry_list, self.clean_list = self.scan_files(data_root_dir)

    def scan_files(self, data_root_path):
        blur_path = Path(data_root_path + "/blur")
        clean_path = Path(data_root_path + "/gt")
        b_list = list(blur_path.glob('*.png'))
        b_list = sorted([str(x) for x in b_list])

        c_list = list(clean_path.glob('*.png'))
        c_list = sorted([str(x) for x in c_list])

        return b_list, c_list

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, index):
        blurry_file_name = self.blurry_list[index]
        clean_img = cv2.imread(self.clean_list[index])[:, :, ::-1]
        blurry_img = cv2.imread(self.blurry_list[index])[:, :, ::-1]

        clean_img = TVTF.to_tensor(clean_img.copy())
        blurry_img = TVTF.to_tensor(blurry_img.copy())
        kernel = torch.zeros(41, 41)

        return clean_img, blurry_img, kernel, blurry_file_name




if __name__ == '__main__':
    dataset = DeblurTrainRealBlurDataset(data_path="../data/real_blur/", mode="J", patch_size=256)
    print(len(dataset))
    print(dataset.blurry_list[100:103])
    print(dataset.clean_list[100:103])
    print(dataset.kernel_list[100:103])
