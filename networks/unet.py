import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Residual Block:
# Two conv2d layers connected with a leakyReLU. The output of the layers is added to the input
class ResBlock(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1):
        super().__init__()
        assert in_chn == out_chn, "In channel must be equal to out channel in conv block."
        self.layers = nn.Sequential(
           nn.Conv2d(in_chn, out_chn, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=kernel_size // 2),
           nn.LeakyReLU(negative_slope=0.2, inplace=True),
           nn.Conv2d(out_chn, out_chn, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=kernel_size // 2),
        )

    def forward(self, x):
        out = self.layers(x) + x
        return out

# Conv Block:
# Allows to combine several residual blocks. Also adds the input to the output
class ConvBlock(nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks):
        super().__init__()
        assert in_chn == out_chn, "In channel must be equal to out channel in conv block."
        self.layers = nn.ModuleList([ResBlock(in_chn, out_chn) for _ in range(num_blocks)])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out + x


class Down(nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks):
        super().__init__()
        self.conv_layers = ConvBlock(in_chn, in_chn, num_blocks=num_blocks)
        self.down_sample = nn.Conv2d(in_chn, out_chn, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        skip = self.conv_layers(x)
        out = self.down_sample(skip)
        return out, skip


class Up(nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks=3):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        self.conv = nn.Conv2d(out_chn * 2, out_chn, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block = ConvBlock(out_chn, out_chn, num_blocks=num_blocks)

    def center_crop(self, feature_map, target_size):
        _, _, height,width = feature_map.size()
        diff_y = (height - target_size[0]) // 2
        diff_x = (width - target_size[1]) // 2
        return feature_map[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, skip):
        up = self.up_sample(x)
        crop_feature_map = self.center_crop(up, skip.shape[2:])
        out = torch.cat([skip, crop_feature_map], 1)
        out = self.conv(out)
        out = self.conv_block(out)
        return out


class UNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=2, nc=64, num_blocks=3, learn_residual=False):
        super().__init__()
        self.learn_residual = learn_residual
        nc = [nc for _ in range(5)]
        self.first_conv = nn.Conv2d(in_nc, nc[0], kernel_size=3, stride=1, padding=1)

        self.down1 = Down(nc[0], nc[1], num_blocks=num_blocks)
        self.down2 = Down(nc[1], nc[2], num_blocks=num_blocks)
        self.down3 = Down(nc[2], nc[3], num_blocks=num_blocks)
        self.down4 = Down(nc[3], nc[4], num_blocks=num_blocks)

        self.mid_conv = ConvBlock(nc[4], nc[4], num_blocks=6)

        self.up4 = Up(nc[-1], nc[-2], num_blocks=num_blocks)
        self.up3 = Up(nc[-2], nc[-3], num_blocks=num_blocks)
        self.up2 = Up(nc[-3], nc[-4], num_blocks=num_blocks)
        self.up1 = Up(nc[-4], nc[-5], num_blocks=num_blocks)

        self.skip_conv1 = nn.Conv2d(nc[1], nc[1], kernel_size=3, stride=1, padding=1)
        self.skip_conv2 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1)
        self.skip_conv3 = nn.Conv2d(nc[3], nc[3], kernel_size=3, stride=1, padding=1)
        self.skip_conv4 = nn.Conv2d(nc[4], nc[4], kernel_size=3, stride=1, padding=1)

        self.final_conv = nn.Sequential(*[
            nn.Conv2d(nc[0], 16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, out_nc, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x, kernel=None):
        out = self.first_conv(x)
        out, skip1 = self.down1(out)
        out, skip2 = self.down2(out)
        out, skip3 = self.down3(out)
        out, skip4 = self.down4(out)

        out = self.mid_conv(out)

        out = self.up4(out, self.skip_conv4(skip4))
        out = self.up3(out, self.skip_conv3(skip3))
        out = self.up2(out, self.skip_conv2(skip2))
        out = self.up1(out, self.skip_conv1(skip1))

        out = self.final_conv(out)
        if self.learn_residual:
            out[:, :3, :, :] = out[:, :3, :, :] + x
        return out


if __name__ == "__main__":
    model = UNet(in_nc=3, out_nc=2, nc=64, num_blocks=6).cuda()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224).cuda()
        y = model(x)
    print(model)
    print(y.shape)

    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))


