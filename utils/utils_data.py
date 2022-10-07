import torch
from .datasets import CamelyonDataset, WSSBDatasetTest

def get_train_dataloaders(camelyon_data_path, patch_size=224, batch_size=16, num_workers=64, val_prop = 0.2, n_samples=None):
    #train_centers=[0,2,4]
    train_centers = [0]

    dataset = CamelyonDataset(camelyon_data_path, train_centers, patch_size=patch_size, n_samples=n_samples)
    len_ds = len(dataset)
    len_val = int(val_prop * len_ds)
    len_train = len_ds - len_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_train, len_val])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader

def get_test_dataloaders(camelyon_data_path, wssb_data_path, patch_size=224, batch_size=1, num_workers=64, n_samples=None):
    #test_centers=[1,3]
    test_centers = [1]
    test_dataset_camelyon = CamelyonDataset(camelyon_data_path, test_centers, patch_size=patch_size, n_samples=n_samples)
    test_dataloader_camelyon = torch.utils.data.DataLoader(test_dataset_camelyon, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset_wssb = WSSBDatasetTest(wssb_data_path)
    test_dataloader_wssb_dict = {}
    for organ in test_dataset_wssb.organ_list:
        test_dataloader_wssb_dict[organ] = torch.utils.data.DataLoader(test_dataset_wssb, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_dataloader_camelyon, test_dataloader_wssb_dict