# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np
from utils.utils_BCD import torch_rgb2od, np_rgb2od
import matplotlib.pyplot as plt
import glob
from datasets.Camelyon_dataset import CamelyonDataset
from datasets.main_dataset import get_dataloaders
from torch.utils.data import DataLoader
from networks.dnet import get

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    Camelyon_path = '/data/BasesDeDatos/Camelyon/Camelyon17/training/patches_224/'
    train_dataset= CamelyonDataset(Camelyon_path, 128)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                                  num_workers=8, pin_memory=True)

    batch=next(iter(train_dataloader))



# NEXT STEP: Knet.py


print('Task failed successfully')

