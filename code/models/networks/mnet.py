import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

def get_model(net_name, fc_hidden_dim=50):
    model = None
    if net_name == "resnet18in":
        model = resnet.ResNet(
                            resnet.BasicBlock, [2, 2, 2, 2], num_classes=fc_hidden_dim, zero_init_residual=False,
                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                            norm_layer=torch.nn.InstanceNorm2d
                            )
    elif net_name == "resnet18bn":
        model = resnet.ResNet(
                            resnet.BasicBlock, [2, 2, 2, 2], num_classes=fc_hidden_dim, zero_init_residual=False,
                            groups=1, width_per_group=64, replace_stride_with_dilation=None,
                            norm_layer=torch.nn.BatchNorm2d
                            )
    elif net_name == "resnet18ft":
        model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_feat = model.fc.in_features
        model.fc = torch.nn.Linear(num_feat, fc_hidden_dim)
    elif net_name == "mobilenetv3s":
        model = torchvision.models.mobilenet_v3_small(weights=None)
        model.classifier = torch.nn.Linear(576, fc_hidden_dim)
    elif net_name == "mobilenetv3sft":
        model = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = torch.nn.Linear(576, fc_hidden_dim)
    else:
        raise Exception("Please set correct net name!")
    return model

class MNet(nn.Module):
    def __init__(self, net_name="resnet18in", stain_dim=3, fc_hidden_dim=50):
        super().__init__()
        self.stain_dim = stain_dim
        self.model = get_model(net_name, fc_hidden_dim)

        #self.model.conv1 = nn.Conv2d(3, 64, stain_dim=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

        self.M_mean = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(fc_hidden_dim, 2*stain_dim),
            #nn.Sigmoid()
        )

        self.M_log_var = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(fc_hidden_dim, 2),
            #nn.Sigmoid()
            #nn.ReLU(inplace=False)
        )

    def forward(self, x):
        # x = rgb_to_yuv(x)  # B x 3 X H x W
        # x = x[:, 0, :, :].unsqueeze(1)  # B x 1 X H x W
        x = F.instance_norm(x)
        x = self.model(x)
        mean = self.M_mean(x) # shape (batch_size, 2*stain_dim)
        mean = mean.view(mean.shape[0], 3, 2) # shape (batch_size, stain_dim, 2)
        l1 = torch.norm(mean, dim=1, keepdim=True) # shape (batch_size, )
        mean = torch.div(mean , l1 + 1e-10) # shape (batch_size, stain_dim, 2)

        var = torch.exp(self.M_log_var(x)) + 1e-10 # shape (batch_size, 2)
        var = var.view(var.shape[0], 1, 2) # shape (batch_size, 1, 2)

        return mean, var

def get_mnet(net_name, stain_dim=3, fc_hidden_dim=50):
    return MNet(net_name, stain_dim, fc_hidden_dim)

