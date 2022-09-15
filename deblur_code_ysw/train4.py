import time
import os
import sys
import torchvision
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils.utils_imgs import AverageMeter
from math import pi, log
from datasets.deconvolution_main_dataset import get_dataloaders
from loss_norm import loss_fn
from networks.dnet import get_dnet
from networks.knet import get_knet
from options4 import set_opts
from torch.distributions import  Normal
import tensorflow as tf

print(torch.__version__)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')
print('using device:', device)




args = set_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))





modes_all = {"center_0": ["train_center_0", "test_center_0"],
             "center_1": ["train_center_1", "test_center_1"],
             "center_2":  ["train_center_2", "test_center_2"],
             "center_3":  ["train_center_3", "test_center_3"]}


def adjust_learning_rate(optimizer, epoch, args):
    if args.run_mode == "center_0":
        if epoch <= 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        elif epoch <= 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6


def get_od_mearsurement_sample(out_DNet,out_KNet1,out_KNet2,batch_size,patch_size):
    p, c, w, h = out_DNet.size()  ######16,2,64,64

    V = torch.zeros(p, 2, 3).to(device)

    rad = torch.normal(mean=torch.zeros([p, 2, 3]), std=1).to(device)  ##########16,2,3

    V[:, :, 0] = out_KNet2
    V[:, :, 1] = out_KNet2
    V[:, :, 2] = out_KNet2

    v = V * rad  ####16,2,3

    M = out_KNet1[:, 0, :, :] + v  #########16,2,3
    M = M.clamp(0, None)

    l = torch.norm(M, dim=2, keepdim=True)
    l_ = 1.0 / (l + 1e-10)
    M = M * l_

    M = M.permute(0, 2, 1)
    C = out_DNet.view(p, 2, w * w)
    od_img = torch.matmul(M, C)  # B,3,4096
    od_img = od_img.view(p, 3, w, w)

    return od_img
def get_od_mearsurement(out_DNet,out_KNet1,out_KNet2,batch_size,patch_size):

    p, c, w, h = out_DNet.size()
    # M=torch.cat((out_KNet1[:, :,0,:],out_KNet2[:, :,0,:]),1)
    M = out_KNet1[:, 0, :, :]
    M=M.permute(0,2,1)
    C=out_DNet.view(p,2,w*w)
    od_img=torch.matmul(M,C) #B,3,4096
    od_img=od_img.view(p,3,w,w)

    return od_img

def get_od_mearsurement_mR(out_DNet, mR,batch_size,patch_size):

    p, c, w, h = out_DNet.size()

    C=out_DNet.view(p,2,w*w)
    mR=mR[:,0,:,:]
    mR=mR.permute(0,2,1)


    od_img=torch.matmul(mR,C) #B,3,4096

    od_img=od_img.view(p,3,w,w)

    return od_img



def main():

    al = args.al
    dnet = get_dnet(args.DNet)
    knet = get_knet(args.KNet, kernel_size=3)
    # if torch.cuda.device_count() > 1:
    #     print("===> Let's use", torch.cuda.device_count(), "GPUs.")
    #     dnet = torch.nn.DataParallel(dnet,device_ids=[1,2])
    #     knet = torch.nn.DataParallel(knet, device_ids=[1,2])
    dnet.to(device)
    knet.to(device)

    # model_dir = './model_alpha1/model_80_before'
    # checkpoint = torch.load(model_dir)
    # knet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)



    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    optimizer_d = optim.Adam(dnet.parameters(), args.lr_D)
    optimizer_k = optim.Adam(knet.parameters(), args.lr_K)

    pretraining_epoch = args.pretraining_epoch
    writer = SummaryWriter(args.log_dir)
    _modes = modes_all[args.run_mode]

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> Loading checkpoint {:s}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.epoch_start = checkpoint['epoch']
            args.step = checkpoint['step']
            args.step_img = checkpoint['step_img']
            optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
            optimizer_k.load_state_dict(checkpoint['optimizer_k_state_dict'])
            dnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)
            knet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)
            step = args.step
            step_img = args.step_img
            print(f'=> Loaded checkpoint {args.resume}')
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args.epoch_start = 0
        step = 0
        step_img = {x: 0 for x in _modes}
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.model_dir, exist_ok=True)

    data_loaders = get_dataloaders(args)
    pre_optimizer_k = optim.Adam(knet.parameters(), lr=5e-4)
    num_iter_per_epoch = {phase: len(data_loaders[phase]) for phase in data_loaders.keys()}


    for epoch in range(args.epoch_start, args.epochs):
        tic = time.time()
        dnet.train()
        knet.train()
        loss_per_epoch = {x: 0 for x in ['Loss','kl_vh_gauss', 'kl_ve_gauss']}
        valid_loss_per_epoch= {x: 0 for x in ['Valid_Loss', 'Valid_kl_vh_gauss', 'Valid_kl_ve_gauss']}
        mse_per_epoch = {x: 0 for x in _modes}
        # 为了让训练更稳定加入一个预训练模糊核的过程
        if epoch < pretraining_epoch:
            pre_kernel_loss = 0
            for ii, data in enumerate(data_loaders[_modes[0]]):

                y, mR  = data[0].to(device), data[1].to(device)

                pre_optimizer_k.zero_grad()
                out_KNet1, out_KNet2 = knet(y)  # B,1,2,3
                kernel_loss , kl_vh_gauss, kl_ve_gauss = loss_fn(out_KNet1, out_KNet2,
                                                         args.alpha_h2, args.alpha_e2, mR,
                                                         )


                kernel_loss.backward()
                pre_optimizer_k.step()
                pre_kernel_loss += kernel_loss.item() / num_iter_per_epoch[_modes[0]]
            print(f"PreTraining: Epoch:{epoch + 1}, L1Loss={pre_kernel_loss:.4e}")
        else:
            adjust_learning_rate(optimizer_k, epoch, args)
            adjust_learning_rate(optimizer_d, epoch, args)
            grad_norm_D = grad_norm_K = 0
            lr_D = optimizer_d.param_groups[0]['lr']
            lr_K = optimizer_k.param_groups[0]['lr']
            phase = _modes[0]

            for ii, data in enumerate(data_loaders[phase]):


                y, mR = data[0].to(device), data[1].to(device)

                optimizer_k.zero_grad()
                optimizer_d.zero_grad()
                out_KNet1,out_KNet2 = knet(y) #B,1,2,3

                out_DNet= dnet(y) #B,2,H,W
                net_output = get_od_mearsurement_sample(out_DNet, out_KNet1, out_KNet2, args.batch_size, args.patch_size)
                residual = criterion(net_output, y)
                print(y.shape)
                print(net_output.shape)
                # net_output_mR = get_od_mearsurement_mR(out_DNet,mR, args.batch_size, args.patch_size)
                # residual_mR = criterion(net_output_mR, y)

                loss,kl_vh_gauss, kl_ve_gauss = loss_fn(out_KNet1,out_KNet2,
                                                            args.alpha_h2,args.alpha_e2, mR
                                                           )

                # losses = loss

                losses=loss*al+(1-al)*residual

                # print(losses)
                losses.backward()
                total_norm_D = nn.utils.clip_grad_norm_(dnet.parameters(), args.clip_grad_D)
                total_norm_K = nn.utils.clip_grad_norm_(knet.parameters(), args.clip_grad_K)
                grad_norm_D = grad_norm_D + total_norm_D / num_iter_per_epoch[phase]
                grad_norm_K = grad_norm_K + total_norm_K / num_iter_per_epoch[phase]
                optimizer_d.step()
                optimizer_k.step()



                loss_per_epoch['Loss'] += losses.item() / num_iter_per_epoch[phase]
                # loss_per_epoch['NegLH'] += neg_lh.item() / num_iter_per_epoch[phase]
                loss_per_epoch['kl_vh_gauss'] += kl_vh_gauss.item() / num_iter_per_epoch[phase]
                loss_per_epoch['kl_ve_gauss'] += kl_ve_gauss.item() / num_iter_per_epoch[phase]

                r_con_e = torch.clamp(out_DNet[:, :1, :, : ].detach().data, 0.0, 1.0)
                r_con_h = torch.clamp(out_DNet[:, 1:, :, :].detach().data, 0.0, 1.0)


                # mse = F.mse_loss(y, net_output)
                mse=residual
                mse_per_epoch[phase] += mse / num_iter_per_epoch[phase]

                if (ii + 1) % args.print_freq == 0:
                    print(f"[Epoch:{epoch + 1:0>4d}/{args.epochs:0>4d}] {phase}: {ii + 1:0>5d}/{num_iter_per_epoch[phase]:0>5d}, "
                          f"loss={losses.item():.4e}, KLvh_gauss={kl_vh_gauss.item():.4e}, KLve_gauss={kl_ve_gauss.item():.4e}, "
                          f"MSE={mse:.4e}, GradNormD:{args.clip_grad_D:.2e}/{total_norm_D:.2e}, "
                          f"GradNormK:{args.clip_grad_K:.2e}/{total_norm_K:.2e}, "
                          f"lrD={lr_D:.1e}, lrK={lr_K:.1e}")

                    writer.add_scalar('Iter Train Loss', losses.item(), step)
                    # writer.add_scalar('Iter Train NegLogLikelihood', neg_lh.item(), step)
                    writer.add_scalar('Iter Train KLvh_gauss', kl_vh_gauss.item(), step)
                    writer.add_scalar('Iter Train KLve_gauss', kl_ve_gauss.item(), step)
                    writer.add_scalar('Iter Train GradientNorm DNet', total_norm_D, step)
                    writer.add_scalar('Iter Train GradientNorm KNet', total_norm_K, step)
                    writer.add_scalar('Iter Train MSE', mse, step)

                    # x1 = torchvision.utils.make_grid(r_con_e, normalize=True, scale_each=True, nrow=args.nrow)
                    # writer.add_image(phase + ' Con_e Image', x1, step)
                    # x2 = torchvision.utils.make_grid(r_con_h, normalize=True, scale_each=True, nrow=args.nrow)
                    # writer.add_image(phase + ' Con_h Image', x2, step)

                    step += 1

            print(f"{phase}: Epoch:{epoch + 1}, Loss={loss_per_epoch['Loss']:.4e},  "
                  f"KLG={loss_per_epoch['kl_vh_gauss']:.4e}, KLDir={loss_per_epoch['kl_ve_gauss']:.4e}, "
                  f"MSE={mse_per_epoch[phase]:.4e}")
            # print(f"m_h={out_KNet1[0, 0, 0, :]}")
            # print(f"m_e={out_KNet2[0, 0, 0, :]}")

            writer.add_scalar('Epoch Train Loss', loss_per_epoch['Loss'], epoch)
            writer.add_scalar('Epoch Train KL vh_gauss', loss_per_epoch['kl_vh_gauss'], epoch)
            writer.add_scalar('Epoch Train KL ve_gauss', loss_per_epoch['kl_ve_gauss'], epoch)
            # writer.add_scalar('Epoch Train NegLH', loss_per_epoch['NegLH'], epoch)
            writer.add_scalar('Epoch Train MSE', mse_per_epoch[phase], epoch)
            writer.add_scalar('Epoch Train Grad Norm DNet', grad_norm_D, epoch)
            writer.add_scalar('Epoch Train Grad Norm KNet', grad_norm_K, epoch)
            writer.add_scalar('Learning rate DNet', lr_D, epoch)
            writer.add_scalar('Learning rate KNet', lr_K, epoch)
        # torch.cuda.empty_cache()
        print('-' * 150)
        if (epoch + 1) >= args.epoch_start_test or ((epoch + 1) % 500 == 0):
            dnet.eval()
            knet.eval()

            psnr_per_iter = {x: [] for x in _modes[1:]}

            mse_per_iter = {x: [] for x in _modes[1:]}

            for phase in _modes[1:]:
                for ii, data in enumerate(data_loaders[phase]):


                    y, mR = data[0].to(device), data[1].to(device)
                    valid_loss = AverageMeter()
                    with torch.no_grad():
                        out_KNet1,out_KNet2 = knet(y)
                        out_DNet = dnet(y)

                        net_output = get_od_mearsurement_sample(out_DNet, out_KNet1, out_KNet2, args.batch_size,
                                                          args.patch_size)
                        valid_residual = criterion(net_output, y)
                        # net_output_mR = get_od_mearsurement_mR(out_DNet, mR, args.batch_size, args.patch_size)
                        # residual_mR = criterion(net_output_mR, y)

                        loss_valid,  kl_vh_gauss_valid, kl_ve_gauss_valid = loss_fn(out_KNet1,out_KNet2,
                                                            args.alpha_h2,args.alpha_e2, mR
                                                                         )
                        # loss = loss_valid
                        loss = loss_valid*al + (1-al)*valid_residual


                        valid_loss.update(loss.data)
                        valid_loss_per_epoch['Valid_Loss'] += loss.item() / num_iter_per_epoch[phase]
                        # valid_loss_per_epoch['Valid_NegLH'] += neg_lh_valid.item() / num_iter_per_epoch[phase]
                        valid_loss_per_epoch['Valid_kl_vh_gauss'] += kl_vh_gauss_valid.item() / num_iter_per_epoch[phase]
                        valid_loss_per_epoch['Valid_kl_ve_gauss'] += kl_ve_gauss_valid.item() / num_iter_per_epoch[phase]





                    r_con_e = torch.clamp(out_DNet[:, :1, :, :].detach().data, 0.0, 1.0)
                    r_con_h = torch.clamp(out_DNet[:, 1:, :, :].detach().data, 0.0, 1.0)

                    # mse = F.mse_loss(y[0,:,:], net_output[0,:,:])
                    mse=valid_residual
                    mse_per_epoch[phase] += mse
                    mse_per_iter[phase].append(mse)

                    writer.add_scalar('Iter Valid Loss', loss.item(), step)
                    # writer.add_scalar('Iter Valid NegLogLikelihood',  neg_lh_valid.item(), step)
                    writer.add_scalar('Iter Valid KLvh_gauss', kl_vh_gauss_valid.item(), step)
                    writer.add_scalar('Iter Valid KLve_gauss', kl_ve_gauss_valid.item(), step)

                    # x1 = torchvision.utils.make_grid(r_con_e, normalize=True, scale_each=True, nrow=args.nrow)
                    # writer.add_image(phase + ' Con_e Image', x1, step)
                    # x2 = torchvision.utils.make_grid(r_con_h, normalize=True, scale_each=True, nrow=args.nrow)
                    # writer.add_image(phase + ' Con_h Image', x2, step)


                mse_per_epoch[phase] /= (ii + 1)

                for k in range(len(psnr_per_iter[phase])):
                    print(f'{phase}, {k + 1:>4d}/{num_iter_per_epoch[phase]:>4d}, mse={mse_per_iter[phase][k]:.4f}')
                print(f'{phase:s}: mse={mse_per_epoch[phase]:.3e}')
                print('-' * 90)
                writer.add_scalar('Test MSE of ' + phase, mse_per_epoch[phase], epoch)
                writer.add_scalar('Epoch Valid Loss', valid_loss_per_epoch['Valid_Loss'], epoch)
                writer.add_scalar('Epoch Valid KL vh_gauss', valid_loss_per_epoch['Valid_kl_vh_gauss'], epoch)
                writer.add_scalar('Epoch Valid KL ve_gauss', valid_loss_per_epoch['Valid_kl_ve_gauss'], epoch)
                # writer.add_scalar('Epoch Valid NegLH', valid_loss_per_epoch['Valid_NegLH'], epoch)


        if (epoch + 1) % args.save_model_freq == 0 or epoch + 1 == args.epochs:
            model_prefix = 'model_'
            save_path_model = os.path.join(args.model_dir, model_prefix + str(epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'step': step + 1,
                'step_img': {x: step_img[x] for x in _modes},
                'deblur_model_state_dict': dnet.state_dict(),
                'kernel_model_state_dict': knet.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'optimizer_k_state_dict': optimizer_k.state_dict(),
            }, save_path_model)
            save_path_deblur_model_state = os.path.join(args.model_dir, 'model_state_d_' + str(epoch + 1))
            torch.save(dnet.state_dict(), save_path_deblur_model_state)
            save_path_kernel_model_state = os.path.join(args.model_dir, 'model_state_k_' + str(epoch + 1))
            torch.save(knet.state_dict(), save_path_kernel_model_state)
        toc = time.time()
        print(f'This epoch take time {toc - tic:.2f} s.')
    writer.close()
    print('Reach the maximal epochs! Finish training')


if __name__ == '__main__':
    main()
