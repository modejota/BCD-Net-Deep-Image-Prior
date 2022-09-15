import torch
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as TVTF
from pathlib import Path
from utils.utils_imgs import npimg_random_crop_pair_patch, pad_kernel


class DeblurTrainTextDataset(Dataset):
    def __init__(self, data_path, patch_size, max_kernel_size):
        super().__init__()
        data_path = Path(data_path)

        clean_list = list(data_path.glob('*orig.png'))
        self.clean_list = sorted([str(x) for x in clean_list])
        blurry_list = list(data_path.glob('*blur.png'))
        self.blurry_list = sorted([str(x) for x in blurry_list])
        kernel_list = list(data_path.glob('*psf.png'))
        self.blur_kernel_list = sorted([str(x) for x in kernel_list])

        self.patch_size = patch_size
        self.real_length = len(self.clean_list)
        self.max_kernel_size = max_kernel_size

    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, index):
        clean_img = cv2.imread(self.clean_list[index])[:, :, ::-1]
        blurry_img = cv2.imread(self.blurry_list[index])[:, :, ::-1]
        blur_kernel = cv2.imread(self.blur_kernel_list[index])[:, :, ::-1]

        clean_img, blurry_img = npimg_random_crop_pair_patch(clean_img, blurry_img, self.patch_size)

        clean_img = TVTF.to_tensor(clean_img.copy())
        blurry_img = TVTF.to_tensor(blurry_img.copy())
        blur_kernel = TVTF.to_tensor(blur_kernel.copy())
        blur_kernel = pad_kernel(blur_kernel, max_size=self.max_kernel_size)[0]  # C x k x k --> k x k
        blur_kernel = torch.div(blur_kernel, torch.sum(blur_kernel))  # sum to 1

        return clean_img, blurry_img, blur_kernel


class DeblurTestTextDataset(Dataset):
    def __init__(self, data_path, max_kernel_size=31, blurry_mode='n_05'):
        super().__init__()
        clean_data_path = Path(data_path + 'orig')
        clean_list = list(clean_data_path.glob('*orig.png'))
        self.clean_list = sorted([str(x) for x in clean_list])
        blur_data_path = Path(data_path + blurry_mode)
        blur_list = list(blur_data_path.glob('*blur.png'))
        self.blurry_list = sorted([str(x) for x in blur_list])
        kernel_data_path = Path(data_path + 'psf')
        kernel_list = list(kernel_data_path.glob('*psf.png'))
        self.kernel_list = sorted([str(x) for x in kernel_list])
        self.max_kernel_size = max_kernel_size

    def __len__(self):
        return  len(self.clean_list)

    def __getitem__(self, index):
        blurry_file_name = self.blurry_list[index]
        clean_img = cv2.imread(self.clean_list[index])[:, :, ::-1]
        blurry_img = cv2.imread(self.blurry_list[index])[:, :, ::-1]
        blur_kernel = cv2.imread(self.kernel_list[index])[:, :, ::-1]

        clean_img = TVTF.to_tensor(clean_img.copy())
        blurry_img = TVTF.to_tensor(blurry_img.copy())
        blur_kernel = TVTF.to_tensor(blur_kernel.copy())
        blur_kernel = pad_kernel(blur_kernel, max_size=self.max_kernel_size)[0]
        blur_kernel = torch.div(blur_kernel, torch.sum(blur_kernel))

        return clean_img, blurry_img, blur_kernel, blurry_file_name


if __name__ == "__main__":
    ds = DeblurTrainTextDataset("../data/text_train_data/", patch_size=256, max_kernel_size=31)
    print(len(ds))