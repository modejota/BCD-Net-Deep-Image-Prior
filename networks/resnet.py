import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
# from utils.utils_imgs import rgb_to_yuv


class ResNet18IN(nn.Module):
    def __init__(self, kernel_size, fc_hidden_dim=50):
        super().__init__()
        self.kernel_size = kernel_size
        self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=fc_hidden_dim, zero_init_residual=False,
                      groups=1, width_per_group=64, replace_stride_with_dilation=None,
                      norm_layer=nn.InstanceNorm2d)

        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.M_mean = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden_dim, 2*kernel_size),
            nn.Sigmoid()
        )

        self.M_var = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Linear(fc_hidden_dim, 2*kernel_size),
            nn.Linear(fc_hidden_dim, 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = rgb_to_yuv(x)  # B x 3 X H x W
        # x = x[:, 0, :, :].unsqueeze(1)  # B x 1 X H x W
        x = F.instance_norm(x)
        x = self.model(x)
        mean = self.M_mean(x)
        mean = mean.view(mean.size()[0],3,2)
        l1 = torch.norm(mean, dim=2, keepdim=True)  # dims= 0 -batch_size, 1- row, 2- cols (norm across)
        l1_ = 1.0 / (l1 + 1e-10)
        # mean = mean * l1_

        var = self.M_var(x)
        var = var.view(var.size()[0],-1,2)

        return mean, var


# class ResNet18BN(nn.Module):
#     def __init__(self, kernel_size, fc_hidden_dim=1000):
#         super().__init__()
#         self.kernel_size = kernel_size
#         self.model = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], num_classes=fc_hidden_dim, zero_init_residual=False,
#                       groups=1, width_per_group=64, replace_stride_with_dilation=None,
#                       norm_layer=nn.BatchNorm2d)
#
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#
#         self.last = nn.Sequential(
#             nn.ReLU(inplace=True),
#             nn.Linear(fc_hidden_dim, kernel_size ** 2),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = rgb_to_yuv(x)  # B x 3 X H x W
#         x = x[:, 0, :, :].unsqueeze(1)  # B x 1 X H x W
#         x = F.instance_norm(x)
#         x = self.model(x)
#         x = self.last(x)
#         x = x.reshape(x.size()[0], 1, self.kernel_size, self.kernel_size)
#         return x




if __name__ == "__main__":

    print(torch.cuda.is_available())
    model = ResNet18IN(kernel_size=3).cuda()

    print(model)
    x = torch.randn(1, 3, 224, 224).cuda()
    y = model(x)
    print(y)
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))