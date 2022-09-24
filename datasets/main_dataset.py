import torch
from torch.utils.data import DataLoader
from datasets.Camelyon_dataset import CamelyonDataset
from torch.utils.data import random_split

def get_dataloaders(args, val_prop = 0.2):
    train_centers=[0,2,4]
    test_centers=[1,3]

    dataset = CamelyonDataset(args.train_data_path,train_centers,patch_size=args.patch_size)
    len_ds = len(dataset)
    len_val = int(args.val_prop * len_ds)
    len_train = len_ds - len_val
    train_dataset, val_dataset = random_split(dataset, [len_train, len_val])
    
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = CamelyonDataset(args.train_data_path, test_centers, patch_size=args.patch_size)
    #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return {"train" : train_dataloader, "val" : val_dataloader, "test" : test_dataloader}

if __name__ == "__main__":
    class args:
        def __init__(self):
            self.batch_size = 1
            self.patch_size = 128
            self.train_data_path = "/data/"
            self.num_workers = 8
    args = args()
    dl = get_dataloaders(args)
    train, test = dl["train"], dl["test"]
    print(len(train.dataset))
    print(len(train))
    print(len(test))