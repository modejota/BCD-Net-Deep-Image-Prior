import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

class MNet(nn.Module):
    def __init__(self, kernel_size=3, fc_hidden_dim=50):
        super().__init__()
        self.kernel_size = kernel_size
        self.model = resnet.ResNet(
                                resnet.BasicBlock, [2, 2, 2, 2], num_classes=fc_hidden_dim, zero_init_residual=False,
                                groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                norm_layer=nn.InstanceNorm2d
                                )

        #self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.M_mean = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(fc_hidden_dim, 2*kernel_size),
            nn.Sigmoid()
        )

        self.M_var = nn.Sequential(
            nn.ReLU(inplace=False),
            #nn.Linear(fc_hidden_dim, 2*kernel_size),
            nn.Linear(fc_hidden_dim, 2),
            nn.Sigmoid()
            #nn.ReLU(inplace=False)
        )

    def forward(self, x):
        # x = rgb_to_yuv(x)  # B x 3 X H x W
        # x = x[:, 0, :, :].unsqueeze(1)  # B x 1 X H x W
        x = F.instance_norm(x)
        x = self.model(x)
        mean = self.M_mean(x) # shape (batch_size, 2*kernel_size)
        mean = mean.view(mean.shape[0], 3, 2)
        l1 = torch.norm(mean, dim=1, keepdim=True) # shape (batch_size, )
        mean = torch.div(mean , l1 + 1e-10)

        var = self.M_var(x) + 1e-10 # shape (batch_size, 2)
        var = var.view(var.shape[0], 1, 2)

        return mean, var

def get_mnet(net_name, kernel_size=3, fc_hidden_dim=50):
    if net_name == "resnet_18_in":
        return MNet(kernel_size)
    elif net_name == "resnet_18_bn":
        # return ResNet18BN(kernel_size)
        raise Exception("Not available yet!")
    else:
        raise Exception("Please set correct net name!")

