import torch
from networks.unet import UNet

def get_cnet(net_name):
    if net_name == 'unet_6':
        return UNet(in_nc=3, out_nc=2, nc=64, num_blocks=6)##############################################

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
        model = get_cnet('unet_2')
        end = time.time()
        print(model)
        print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))