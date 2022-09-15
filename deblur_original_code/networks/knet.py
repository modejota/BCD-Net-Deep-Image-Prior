import torch
from networks.resnet import ResNet18IN, ResNet18BN


def get_knet(net_name, kernel_size):
    if net_name == "resnet_18_in":
        return ResNet18IN(kernel_size)
    elif net_name == "resnet_18_bn":
        return ResNet18BN(kernel_size)
    else:
        raise Exception("Please set correct net name!")


if __name__ == "__main__":
    import time
    model = get_knet(net_name="resnet_18_bn", kernel_size=31).cuda()
    x = torch.rand(1, 3, 256, 256).cuda()
    tic = time.time()
    y = model(x)
    end = time.time()
    print(end - tic)
    print(y.shape)
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))

