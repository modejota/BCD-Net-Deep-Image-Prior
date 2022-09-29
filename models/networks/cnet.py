from .unet import UNet

def get_cnet(net_name):
    if net_name == 'unet_9':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=9)
    elif net_name == 'unet_6':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=6)
    elif net_name == 'unet_5':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=5)
    elif net_name == 'unet_4':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=4)
    elif net_name == 'unet_3':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=3)
    elif net_name == 'unet_2':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=2)
    elif net_name == 'unet_1_32':
        return UNet(in_nc=3, out_nc=6, nc=32, num_blocks=1)
    else:
        raise Exception('Invalid net')
