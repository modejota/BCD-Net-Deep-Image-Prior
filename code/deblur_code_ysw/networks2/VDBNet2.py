import torch
import torch.nn as nn

from networks.cnet import get_dnet
from networks.mnet import get_knet


class VDBNet(nn.Module):
    def __init__(self, DeblurNetModel, KNetModel, blur_kernel_size, dirichlet_para_stretch,
                 pretraining_epoch):
        super().__init__()
        self.DNet = get_dnet(DeblurNetModel)
        self.KNet = get_knet(KNetModel, kernel_size=blur_kernel_size)
        self.kernel_size = blur_kernel_size
        self.dirichlet_para_stretch = dirichlet_para_stretch
        self.pretraining_epoch = pretraining_epoch

    def forward(self, x, cur_epoch, mode='train'):
        batch_size, x_channel, x_H, x_W = x.size()
        if mode.lower() == 'train':
            out_KNet = self.KNet(x)  # B x 1 x k x k
            xi = out_KNet.reshape(batch_size, -1) * self.dirichlet_para_stretch + 1e-7
            q_k = torch.distributions.dirichlet.Dirichlet(xi)
            kernel_sample_return = q_k.rsample()
            kernel_sample_return = kernel_sample_return.reshape(batch_size, 1, self.kernel_size, -1)
            if cur_epoch < self.pretraining_epoch:
                out_DNet = torch.zeros_like(x)
                return out_DNet, out_KNet, kernel_sample_return
            else:
                out_DNet = self.DNet(x, kernel_sample_return)
                return out_DNet, out_KNet, kernel_sample_return

        elif mode.lower() == 'test':
            out_KNet = self.KNet(x)  # B x 1 x k x k
            dirichlet_mode = out_KNet
            kernel = dirichlet_mode.reshape(dirichlet_mode.size()[0], 1, self.kernel_size, self.kernel_size)
            out_DNet = self.DNet(x, kernel)
            return out_DNet, out_KNet, kernel



if __name__ == "__main__":
    model = VDBNet(DeblurNetModel='unet_sft_6', KNetModel='resnet_18', blur_kernel_size=31,
                 dirichlet_para_stretch=5000, pretraining_epoch=0).cuda()
    x = torch.rand(1, 3, 256, 256).cuda()
    out_DNet, out_KNet, kernel = model(x)
    print(model)
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))
