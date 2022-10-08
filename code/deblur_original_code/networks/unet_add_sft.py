import torch
from torch import nn
import torch.nn.functional as F

class SFT_Layer(nn.Module):
    def __init__(self, nf=64, code_len=20):
        """
        :param nf: input feature map channel
        :param code_len: blur kernel code length
        BCHW --> BCHW
        """
        super().__init__()
        self.mul_conv1 = nn.Conv2d(code_len + nf, 32, kernel_size=3, stride=1, padding=1)
        self.mul_leaky = nn.LeakyReLU(0.2)
        self.mul_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

        self.add_conv1 = nn.Conv2d(code_len + nf, 32, kernel_size=3, stride=1, padding=1)
        self.add_leaky = nn.LeakyReLU(0.2)
        self.add_conv2 = nn.Conv2d(32, nf, kernel_size=3, stride=1, padding=1)

    def forward(self, feature_maps, kernel_maps):
        cat_input = torch.cat((feature_maps, kernel_maps), dim=1)
        mul = torch.sigmoid(self.mul_conv2(self.mul_leaky(self.mul_conv1(cat_input))))
        add = self.add_conv2(self.add_leaky(self.add_conv1(cat_input)))
        return feature_maps * mul + add


class Kernel2CodeNet(nn.Module):
    """
    blur kernel --> blur vector
    """
    def __init__(self, in_nc=1, nf=64, code_len=10):
        super().__init__()
        self.ConvNet = nn.Sequential(*[
            nn.Conv2d(in_nc, nf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, code_len, kernel_size=3, stride=1, padding=1),
        ])
        self.globalPooling = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, input):
        # input: B x 1 x K x K
        conv = self.ConvNet(input) # B x code_len x K x K
        flat = self.globalPooling(conv) # B x C x 1 x 1
        return flat.view(flat.size()[:2]) # torch size: [B, code_len]


class ResBlock(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1):
        super().__init__()
        self.layers = nn.Sequential(*[
           nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(out_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        ])

    def forward(self, x):
        out = self.layers(x) + x
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_chn, out_chn,  num_blocks=3):
        super().__init__()
        assert in_chn == out_chn
        self.layers = nn.ModuleList([ResBlock(in_chn, out_chn) for _ in range(num_blocks)])

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
        return out + x


class Down(nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks=3, code_len=20):
        super().__init__()
        self.conv_layers = ConvBlock(in_chn, in_chn, num_blocks=num_blocks)
        self.sft_layer = SFT_Layer(nf=in_chn, code_len=code_len)
        self.down_sample = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=2, padding=1)

    def forward(self, x, kernel_code):
        B, C, H, W = x.size()
        B_h, C_h = kernel_code.size()
        ker_code_exp = kernel_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel_map stretch
        x = self.conv_layers(x)
        skip = self.sft_layer(x, ker_code_exp)
        x = self.down_sample(skip)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_chn, out_chn, num_blocks=3, code_len=20):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chn, out_chn, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv = nn.Conv2d(out_chn * 2, out_chn, kernel_size=3, stride=1, padding=1)
        self.sft_layer = SFT_Layer(nf=out_chn, code_len=code_len)
        self.conv_block = ConvBlock(out_chn, out_chn, num_blocks=num_blocks)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, skip, kernel_code):    
        up = self.up(x)
        crop1 = self.center_crop(up, skip.shape[2:])
        out = torch.cat([skip, crop1], 1)
        out = self.conv(out)
        B, C, H, W = out.size()
        B_h, C_h = kernel_code.size()
        ker_code_exp = kernel_code.view((B_h, C_h, 1, 1)).expand((B_h, C_h, H, W))  # kernel_map stretch
        out = self.sft_layer(out, ker_code_exp)
        out = self.conv_block(out)
        return out


class UNetAddSFT(nn.Module):
    def __init__(self, in_nc=3, out_nc=6, nc=64, num_blocks=3, learn_residual=True):
        super().__init__()
        self.learn_residual = learn_residual
        code_len = 20
        nc = [nc for _ in range(5)]
        self.kernel2codenet = Kernel2CodeNet(code_len=code_len)

        self.first_conv = nn.Conv2d(in_nc, nc[0], kernel_size=7, stride=1, padding=3)

        self.down1 = Down(nc[0], nc[1], num_blocks=num_blocks, code_len=code_len)
        self.down2 = Down(nc[1], nc[2], num_blocks=num_blocks, code_len=code_len)
        self.down3 = Down(nc[2], nc[3], num_blocks=num_blocks, code_len=code_len)
        self.down4 = Down(nc[3], nc[4], num_blocks=num_blocks, code_len=code_len)

        self.mid_conv = ConvBlock(nc[4], nc[4], num_blocks=5)

        self.up4 = Up(nc[-1], nc[-2], num_blocks=num_blocks, code_len=code_len)
        self.up3 = Up(nc[-2], nc[-3], num_blocks=num_blocks, code_len=code_len)
        self.up2 = Up(nc[-3], nc[-4], num_blocks=num_blocks, code_len=code_len)
        self.up1 = Up(nc[-4], nc[-5], num_blocks=num_blocks, code_len=code_len)

        self.skip_conv1 = nn.Conv2d(nc[1], nc[1], kernel_size=3, stride=1, padding=1)
        self.skip_conv2 = nn.Conv2d(nc[2], nc[2], kernel_size=3, stride=1, padding=1)
        self.skip_conv3 = nn.Conv2d(nc[3], nc[3], kernel_size=3, stride=1, padding=1)
        self.skip_conv4 = nn.Conv2d(nc[4], nc[4], kernel_size=3, stride=1, padding=1)


        self.final_conv = nn.Sequential(*[
            nn.Conv2d(nc[0], 16, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(16, out_nc, kernel_size=7, stride=1, padding=3),
        ])


    def forward(self, x, kernel=None):
        kernel_code = self.kernel2codenet(kernel)
        
        out = self.first_conv(x)
        out, skip1 = self.down1(out, kernel_code)
        out, skip2 = self.down2(out, kernel_code)
        out, skip3 = self.down3(out, kernel_code)
        out, skip4 = self.down4(out, kernel_code)


        out = self.mid_conv(out)

        out = self.up4(out, self.skip_conv4(skip4), kernel_code)
        out = self.up3(out, self.skip_conv3(skip3), kernel_code)
      
        out = self.up2(out, self.skip_conv2(skip2), kernel_code)
        out = self.up1(out, self.skip_conv1(skip1), kernel_code)

        out = self.final_conv(out)
        # if self.learn_residual:
        #     out[:, :3, :, :] = out[:, :3, :, :] + x

        return out



if __name__ == "__main__":
    model = UNetAddSFT(in_nc=3, out_nc=6, nc=64, num_blocks=3).cuda()
    print(model)
    x = torch.rand(2, 3, 180, 160).cuda()
    k = torch.rand(2, 1, 21, 21).cuda()
    y = model(x, k)
    print(y.shape)
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))