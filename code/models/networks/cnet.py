from .unet import UNet

def get_cnet(net_name):
    if net_name == 'unet9':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=9)
    elif net_name == 'unet6':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=6)
    elif net_name == 'unet5':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=5)
    elif net_name == 'unet4':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=4)
    elif net_name == 'unet3':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=3)
    elif net_name == 'unet2':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=2)
    else:
        raise Exception('Invalid net')
