import torch
from .datasets import CamelyonDataset




def get_train_dataloaders(data_path, patch_size=128, batch_size=16, num_workers=8, val_prop = 0.2, n_samples=None):
    #train_centers=[0,2,4]
    train_centers = [0]
    #test_centers=[1,3]
    test_centers = [1]

    dataset = CamelyonDataset(data_path, train_centers, patch_size=patch_size, n_samples=n_samples)
    len_ds = len(dataset)
    len_val = int(val_prop * len_ds)
    len_train = len_ds - len_val
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len_train, len_val])
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader

def get_test_dataloader(data_path, patch_size=128, batch_size=16, num_workers=8, n_samples=None):
    test_centers = [1]
    test_dataset = CamelyonDataset(data_path, test_centers, patch_size=patch_size, n_samples=n_samples)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    return test_dataloader