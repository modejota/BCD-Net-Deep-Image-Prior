import torch
from networks.resnet import ResNet18IN #,ResNet18BN

def get_mnet(net_name, kernel_size):
    if net_name == "resnet_18_in":
        return ResNet18IN(kernel_size)
    elif net_name == "resnet_18_bn":
        # return ResNet18BN(kernel_size)
        raise Exception("Not available yet!")
    else:
        raise Exception("Please set correct net name!")

