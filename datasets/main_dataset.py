from torch.utils.data import DataLoader
from datasets.Camelyon_dataset import CamelyonDataset
from torch.utils.data import random_split

def get_dataloaders(data_path, patch_size=128, batch_size=16, num_workers=8, val_prop = 0.2):
    #train_centers=[0,2,4]
    train_centers = [0]
    #test_centers=[1,3]
    test_centers = [1]

    dataset = CamelyonDataset(data_path, train_centers, patch_size=patch_size)
    len_ds = len(dataset)
    len_val = int(val_prop * len_ds)
    len_train = len_ds - len_val
    train_dataset, val_dataset = random_split(dataset, [len_train, len_val])
    
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_dataset = CamelyonDataset(data_path, test_centers, patch_size=patch_size)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    return {"train" : train_dataloader, "val" : val_dataloader, "test" : test_dataloader}
