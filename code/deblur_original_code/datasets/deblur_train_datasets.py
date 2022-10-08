import random
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import torchvision.transforms.functional as TVTF
from utils.utils_imgs import npimg_random_crop_patch


class DeblurTrainDataset(Dataset):
    def __init__(self, data_path, patch_size):
        super().__init__()
        data_path = Path(data_path)
        data_list = list(data_path.glob('*.jpg')) + list(data_path.glob('*.png')) + list(data_path.glob('*.bmp'))
        self.data_list = sorted([str(x) for x in data_list])
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        clean_img = cv2.imread(self.data_list[index])[:,:,::-1]
        clean_img = npimg_random_crop_patch(clean_img, self.patch_size)
        clean_img = TVTF.to_tensor(clean_img.copy())
        return clean_img


if __name__ == "__main__":
    dataset = DeblurTrainDataset("../data/open_val", 100)
    from torch.utils.data import DataLoader
    dataloder = DataLoader(dataset, batch_size=1)
    min_H = 999999
    min_W = 999999
    for ii, data in enumerate(dataloder):
        im = data
        H, W = im.shape[2], im.shape[3]
        min_H = min(min_H, H)
        min_W = min(min_W, W)
        print(ii + 1, H, W)
    print(f"Min H: {min_H}")
    print(f"Min W: {min_W}")



