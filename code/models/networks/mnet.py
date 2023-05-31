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


class AnalyticalMNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Y_OD, C_mean, M_ref, sigma_sq, lambda_sq):
        """ Summary of forward function.
        Args:
            Y_OD (batch_size, 3, H, W): input OD image
            C_mean (batch_size, n_stains, H, W): output of CNet
            M_ref (batch_size, stain_dim, n_stains): reference stain matrix
            sigma_sq (float): variance of color prior
            lambda_sq (float): variance of observation model
        Returns:
            out_mean (batch_size, stain_dim, n_stains): _description_
            out_var (batch_size, n_stains, n_stains): _description_
        """

        batch_size, n_stains = C_mean.shape[0], C_mean.shape[1]

        inv_lambda_sq = 1.0 / lambda_sq
        inv_sigma_sq = 1.0 / sigma_sq

        C_flat = C_mean.view(batch_size, n_stains, -1) # shape (batch_size, n_stains, H*W)

        out_var_inv = inv_sigma_sq * torch.eye(n_stains).to(C_flat.device) + inv_lambda_sq * C_flat @ C_flat.transpose(1, 2) # shape (batch_size, n_stains, n_stains)
        out_var = torch.inverse(out_var_inv) # shape (batch_size, n_stains, n_stains)
        out_mean = (inv_sigma_sq * M_ref + inv_lambda_sq * Y_OD.transpose(1, 2) @ C_flat.transpose(1, 2) ) @ out_var # shape (batch_size, stain_dim, n_stains)

        return out_mean, out_var


