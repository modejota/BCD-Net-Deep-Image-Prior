import torch
from torch.utils.data import DataLoader

from datasets.deconvolution_center_dataset import DeconvolutionTrainDataset, DeconvolutionTestDataset


def get_dataloaders(args):

    if args.run_mode == "center_0":
        train_dataset = DeconvolutionTrainDataset(data_path=args.data_center_path, patch_size=args.patch_size, augment=False, add_noise=False)
        test_dataset = DeconvolutionTestDataset(data_path=args.data_center_test_path)


        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

        return {"train_center_0": train_dataloader, "test_center_0": test_dataloader}


if __name__ == "__main__":
    class args:
        def __init__(self):
            self.batch_size = 1
            self.patch_size = 128
            self.data_gopro_path = "/data/"
            self.run_mode = "center_0"
            self.num_workers = 8
    args = args()
    dl = get_dataloaders(args)
    train, test = dl["train_center_0"], dl["test_center_0"]
    print(len(train.dataset))
    print(len(train))
    print(len(test))
