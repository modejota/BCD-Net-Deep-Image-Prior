import torch
from torch.utils.data import DataLoader
from datasets.Camelyon_dataset import CamelyonDataset

def get_dataloaders(args):
    train_centers=[0,2,4]
    test_centers=[1,3]
    train_dataset= CamelyonDataset(args.train_data_path,train_centers,patch_size=args.patch_size)

    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    test_dataset = CamelyonDataset(args.train_data_path,test_centers, patch_size=args.patch_size)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

    return {"Train" : train_dataloader, "Test" : test_dataloader}

if __name__ == "__main__":
    class args:
        def __init__(self):
            self.batch_size = 1
            self.patch_size = 128
            self.train_data_path = "/data/"
            self.num_workers = 8
    args = args()
    print('working')
    dl = get_dataloaders(args)
    train, test = dl["Train"], dl["Test"]
    print(len(train.dataset))
    print(len(train))
    print(len(test))

    for data in train:
        imgs, matrixes= data
        print(imgs.shape) #batch_size, 3, patch_size, patch_size
        print(matrixes.shape) #batch_size, 1, 3, n_stains=2
