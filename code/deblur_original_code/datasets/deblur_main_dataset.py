import torch
from torch.utils.data import DataLoader
from datasets.deblur_realblur_dataset import DeblurTrainRealBlurDataset, DeblurTestRealBlurDataset
from datasets.deblur_gopro_dataset import DeblurTrainGoProDataset, DeblurTestGoProDataset
from datasets.deblur_text_dataset import DeblurTrainTextDataset, DeblurTestTextDataset

def get_dataloaders(args):
    if args.run_mode == "Text":
        train_dataset = DeblurTrainTextDataset(args.train_data_text_path, patch_size=args.patch_size, max_kernel_size=args.max_size)
        test_dataset = DeblurTestTextDataset(args.test_data_text_path, max_kernel_size=args.max_size)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

        return {"train_Text" : train_dataloader, "test_Text": test_dataloader}

    elif args.run_mode == "GoPro":
        train_dataset = DeblurTrainGoProDataset(data_path=args.data_gopro_path, patch_size=args.patch_size, augment=True, add_noise=False)
        test_dataset = DeblurTestGoProDataset(data_path=args.data_gopro_path)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

        return {"train_GoPro": train_dataloader, "test_GoPro": test_dataloader}

    elif args.run_mode == "RealBlur_J":
        train_dataset = DeblurTrainRealBlurDataset(args.data_realblur_path, patch_size=args.patch_size, mode='J', augment=True)
        test_dataset = DeblurTestRealBlurDataset(args.data_realblur_path, mode='J')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

        return {"train_RealBlur_J": train_dataloader, "test_RealBlur_J": test_dataloader}

    elif args.run_mode == "RealBlur_R":
        train_dataset = DeblurTrainRealBlurDataset(args.data_realblur_path, patch_size=args.patch_size, mode='R', augment=True)
        test_dataset = DeblurTestRealBlurDataset(args.data_realblur_path, mode='R')

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                     num_workers=args.num_workers, pin_memory=True)

        return {"train_RealBlur_R": train_dataloader, "test_RealBlur_R": test_dataloader}

    else:
        raise Exception("Invalid Dataset Name!")


def get_datasets(args):
    if args.run_mode == "Text":
        train_dataset = DeblurTrainTextDataset(args.train_data_text_path, patch_size=args.patch_size, max_kernel_size=args.max_size)
        test_dataset = DeblurTestTextDataset(args.test_data_text_path, max_kernel_size=args.max_size)

        return {"train_Text" : train_dataset, "test_Text": test_dataset}

    elif args.run_mode == "GoPro":
        train_dataset = DeblurTrainGoProDataset(data_path=args.data_gopro_path, patch_size=args.patch_size, augment=True, add_noise=False)
        test_dataset = DeblurTestGoProDataset(data_path=args.data_gopro_path)

        return {"train_GoPro": train_dataset, "test_GoPro": test_dataset}

    elif args.run_mode == "RealBlur_J":
        train_dataset = DeblurTrainRealBlurDataset(args.data_realblur_path, patch_size=args.patch_size, mode='J', augment=True)
        test_dataset = DeblurTestRealBlurDataset(args.data_realblur_path, mode='J')

        return {"train_RealBlur_J": train_dataset, "test_RealBlur_J": test_dataset}

    elif args.run_mode == "RealBlur_R":
        train_dataset = DeblurTrainRealBlurDataset(args.data_realblur_path, patch_size=args.patch_size, mode='R', augment=True)
        test_dataset = DeblurTestRealBlurDataset(args.data_realblur_path, mode='R')

        return {"train_RealBlur_R": train_dataset, "test_RealBlur_R": test_dataset}


    else:
        raise Exception("Invalid Dataset Name!")


if __name__ == "__main__":
    class args:
        def __init__(self):
            self.batch_size = 4
            self.patch_size = 256
            self.data_gopro_path = "../data/gopro_patch_kernels/"
            self.run_mode = "GoPro"
            self.num_workers = 8
    args = args()
    dl = get_dataloaders(args)
    train, test = dl["train_GoPro"], dl["test_GoPro"]
    print(len(train.dataset))
    print(len(train))
    print(len(test))
