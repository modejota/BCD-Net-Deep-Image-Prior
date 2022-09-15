import torch
from networks.unet import UNet
from networks.unet_add_sft import UNetAddSFT
from networks.Unet_seg import DynamicUNet
def get_dnet(net_name):
    if net_name == 'unet_sft_9':
        return UNetAddSFT(in_nc=3, out_nc=6, nc=64, num_blocks=9)

    elif net_name == 'unet_sft_6':
        return UNetAddSFT(in_nc=3, out_nc=6, nc=64, num_blocks=6)

    elif net_name == 'unet_sft_5':
        return UNetAddSFT(in_nc=3, out_nc=6, nc=64, num_blocks=5)

    elif net_name == 'unet_sft_4':
        return UNetAddSFT(in_nc=3, out_nc=6, nc=64, num_blocks=4)

    elif net_name == 'unet_sft_3':
        return UNetAddSFT(in_nc=3, out_nc=6, nc=64, num_blocks=3)

    elif net_name == 'unet_sft_2':
        return UNetAddSFT(in_nc=3, out_nc=2, nc=64, num_blocks=2)############channels=8

    elif net_name == 'unet_9':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=9)

    elif net_name == 'unet_6':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=6)##############################################
        # return UNet(in_channels=3, out_channels=2, depth=4, wf=64, slope=0.2)
        # return DynamicUNet(filters = [16,32,64,128,256],input_channels=3, output_channels=2)
    elif net_name == 'unet_5':
        return UNet(in_nc=3, out_nc=6, nc=64, num_blocks=5)

    elif net_name == 'unet_4':
        return UNet(in_nc=3, out_nc=6, nc=64, num_blocks=4)

    elif net_name == 'unet_3':
        return UNet(in_nc=3, out_nc=6, nc=64, num_blocks=3)

    elif net_name == 'unet_2':
        return UNet(in_nc=3, out_nc=6, nc=64, num_blocks=2)

    elif net_name == 'unet_1_32':
        return UNet(in_nc=3, out_nc=6, nc=32, num_blocks=1)


    else:
        raise Exception('Invalid net')


if __name__ == "__main__":
    # import time
    # torch.cuda.empty_cache()
    # model = deblur_net('syn_with_kernel').cuda()
    # x = torch.rand(1, 3, 128, 72).cuda()
    # k = torch.rand(1, 1, 31, 31).cuda()
    # tic = time.time()
    # with torch.no_grad():
    #     y = model(x, k)
    # end = time.time()
    # print(end - tic)
    # print(model)
    # print(y.shape)
    # print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))
    import time
    torch.cuda.empty_cache()
    model = get_dnet('unet_2')
    end = time.time()
    print(model)
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))


