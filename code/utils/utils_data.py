import torch
from .datasets import CamelyonDataset, WSSBDatasetTest, FilesDataset

def get_train_dataloaders(camelyon_data_path, patch_size=224, batch_size=16, num_workers=64, val_prop = 0.2, n_samples=None, train_centers=[0,2,4]):
    dataset = CamelyonDataset(camelyon_data_path, train_centers, patch_size=patch_size, n_samples=n_samples)
    len_ds = len(dataset)
    len_val = int(val_prop * len_ds)
    len_train = len_ds - len_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_train, len_val], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if len(val_dataset) < batch_size:
        batch_size = len(val_dataset)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader

def get_camelyon_test_dataloader(camelyon_data_path, patch_size=None, num_workers=64, n_samples=None, test_centers=[1,3]):
    test_dataset_camelyon = CamelyonDataset(camelyon_data_path, test_centers, patch_size=patch_size, n_samples=n_samples)
    test_dataloader_camelyon = torch.utils.data.DataLoader(test_dataset_camelyon, batch_size=1, shuffle=False, num_workers=num_workers)
    return test_dataloader_camelyon

def get_wssb_dataloader(wssb_data_path, num_workers=64):
    
    test_dataset_wssb = WSSBDatasetTest(wssb_data_path, organ_list=['Lung', 'Breast', 'Colon'])
    test_dataloader_wssb = torch.utils.data.DataLoader(test_dataset_wssb, batch_size=1, shuffle=False, num_workers=num_workers)
    return test_dataloader_wssb

def get_wssb_dataloader_dict(wssb_data_path, num_workers=64):
    
    test_dataloader_wssb_dict = {}
    for organ in ['Lung', 'Breast', 'Colon']:
        test_dataset_wssb = WSSBDatasetTest(wssb_data_path, organ_list=[organ])
        test_dataloader_wssb_dict[organ] = torch.utils.data.DataLoader(test_dataset_wssb, batch_size=1, shuffle=False, num_workers=num_workers)

    return test_dataloader_wssb_dict

def get_files_dataloader(files_path, patch_size=None, batch_size = 16, num_workers=64):
    dataset = FilesDataset(files_path, patch_size=patch_size)
    if len(dataset) < batch_size:
        batch_size = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dataloader