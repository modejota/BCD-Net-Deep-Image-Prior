from .unet import UNet
from .unet_add_sft import UNetAddSFT

def get_cnet(net_name):
    if net_name[-3:] == 'sft':
        net_name = net_name[:-3]
        num_blocks = int(net_name[-1])
        return UNetAddSFT(in_nc=3, out_nc=2, nc=64, num_blocks=num_blocks)
    else:
        num_blocks = int(net_name[-1])
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=num_blocks)
