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
from networks.cnet import get_cnet
from networks.mnet import get_mnet
from options3 import set_opts
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


def get_od_mearsurement_sample(out_CNet,out_MNet1,out_MNet2,batch_size,patch_size):
    p, c, w, h = out_CNet.size()  ######16,2,64,64

    V = torch.zeros(p, 2, 3).to(device)

    rad = torch.normal(mean=torch.zeros([p, 2, 3]), std=1).to(device)  ##########16,2,3

    V[:, :, 0] = out_MNet2
    V[:, :, 1] = out_MNet2
    V[:, :, 2] = out_MNet2

    v = V * rad  ####16,2,3

    M = out_MNet1[:, 0, :, :] + v  #########16,2,3
    M = M.clamp(0, None)

    l = torch.norm(M, dim=2, keepdim=True)
    l_ = 1.0 / (l + 1e-10)
    M = M * l_

    M = M.permute(0, 2, 1)
    C = out_CNet.view(p, 2, w * w)
    od_img = torch.matmul(M, C)  # B,3,4096
    od_img = od_img.view(p, 3, w, w)

    return od_img
def get_od_mearsurement(out_CNet,out_MNet1,out_MNet2,batch_size,patch_size):

    p, c, w, h = out_CNet.size()
    # M=torch.cat((out_KNet1[:, :,0,:],out_KNet2[:, :,0,:]),1)
    M = out_MNet1[:, 0, :, :]
    M=M.permute(0,2,1)
    C=out_CNet.view(p,2,w*w)
    od_img=torch.matmul(M,C) #B,3,4096
    od_img=od_img.view(p,3,w,w)

    return od_img

def get_od_mearsurement_mR(out_CNet, mR,batch_size,patch_size):

    p, c, w, h = out_CNet.size()

    C=out_CNet.view(p,2,w*w)
    mR=mR[:,0,:,:]
    mR=mR.permute(0,2,1)


    od_img=torch.matmul(mR,C) #B,3,4096

    od_img=od_img.view(p,3,w,w)

    return od_img



def main():

    al = args.al
    pre_kl=args.pre_kl
    pre_mse = args.pre_mse
    cnet = get_cnet(args.CNet)
    mnet = get_mnet(args.MNet, kernel_size=3)
    # if torch.cuda.device_count() > 1:
    #     print("===> Let's use", torch.cuda.device_count(), "GPUs.")
    #     dnet = torch.nn.DataParallel(dnet,device_ids=[1,2])
    #     knet = torch.nn.DataParallel(knet, device_ids=[1,2])
    cnet.to(device)
    mnet.to(device)



    criterion = nn.MSELoss(reduction='sum')
    criterion.cuda()

    optimizer_c = optim.Adam(cnet.parameters(), args.lr_C)
    optimizer_m = optim.Adam(mnet.parameters(), args.lr_M)

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
            optimizer_c.load_state_dict(checkpoint['optimizer_c_state_dict'])
            optimizer_m.load_state_dict(checkpoint['optimizer_m_state_dict'])
            cnet.load_state_dict(checkpoint['deblur_model_state_dict'], strict=True)
            mnet.load_state_dict(checkpoint['kernel_model_state_dict'], strict=True)
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
    pre_optimizer_m = optim.Adam(mnet.parameters(), lr=5e-4)
    pre_optimizer_c = optim.Adam(cnet.parameters(), lr=5e-4)
    num_iter_per_epoch = {phase: len(data_loaders[phase]) for phase in data_loaders.keys()}


    for epoch in range(args.epoch_start, args.epochs):
        tic = time.time()
        cnet.train()
        mnet.train()
        loss_per_epoch = {x: 0 for x in ['Loss','kl_vh_gauss', 'kl_ve_gauss']}
        valid_loss_per_epoch= {x: 0 for x in ['Valid_Loss', 'Valid_kl_vh_gauss', 'Valid_kl_ve_gauss']}
        mse_per_epoch = {x: 0 for x in _modes}
        # 为了让训练更稳定加入一个预训练模糊核的过程
        if epoch < pretraining_epoch:
            pre_train_loss = 0
            for ii, data in enumerate(data_loaders[_modes[0]]):

                y, mR  = data[0].to(device), data[1].to(device)

                pre_optimizer_m.zero_grad()
                pre_optimizer_c.zero_grad()
                out_MNet1, out_MNet2 = mnet(y)  # B,1,2,3

                out_CNet = cnet(y)  # B,2,H,W
                net_output = get_od_mearsurement_sample(out_CNet, out_MNet1, out_MNet2, args.batch_size,
                                                        args.patch_size)
                pre_residual = criterion(net_output, y)

                pre_loss, pre_kl_vh_gauss, pre_kl_ve_gauss = loss_fn(out_MNet1, out_MNet2,
                                                         args.alpha_h2, args.alpha_e2, mR
                                                         )
                # pre_losses = pre_loss
                pre_losses = pre_loss * pre_kl + pre_mse * pre_residual
                pre_losses.backward()
                pre_optimizer_m.step()
                pre_optimizer_c.step()
                pre_train_loss += pre_losses.item() / num_iter_per_epoch[_modes[0]]
            print(f"PreTraining: Epoch:{epoch + 1}, L1Loss={pre_train_loss:.4e}")
        else:
            adjust_learning_rate(optimizer_m, epoch, args)
            adjust_learning_rate(optimizer_c, epoch, args)
            grad_norm_C = grad_norm_M= 0
            lr_C = optimizer_c.param_groups[0]['lr']
            lr_M = optimizer_m.param_groups[0]['lr']
            phase = _modes[0]

            for ii, data in enumerate(data_loaders[phase]):


                y, mR = data[0].to(device), data[1].to(device)

                optimizer_m.zero_grad()
                optimizer_c.zero_grad()
                out_MNet1,out_MNet2 = mnet(y) #B,1,2,3

                out_CNet= cnet(y) #B,2,H,W
                net_output = get_od_mearsurement_sample(out_CNet, out_MNet1, out_MNet2, args.batch_size, args.patch_size)
                residual = criterion(net_output, y)


                loss,kl_vh_gauss, kl_ve_gauss = loss_fn(out_MNet1,out_MNet2,
                                                            args.alpha_h2,args.alpha_e2, mR
                                                           )

                # losses = loss
                losses = loss * al + (1 - al) * residual
                losses.backward()
                total_norm_C = nn.utils.clip_grad_norm_(cnet.parameters(), args.clip_grad_C)
                total_norm_M = nn.utils.clip_grad_norm_(mnet.parameters(), args.clip_grad_M)
                grad_norm_C = grad_norm_C + total_norm_C / num_iter_per_epoch[phase]
                grad_norm_M = grad_norm_M + total_norm_M / num_iter_per_epoch[phase]
                optimizer_c.step()
                optimizer_m.step()



                loss_per_epoch['Loss'] += losses.item() / num_iter_per_epoch[phase]
                # loss_per_epoch['NegLH'] += neg_lh.item() / num_iter_per_epoch[phase]
                loss_per_epoch['kl_vh_gauss'] += kl_vh_gauss.item() / num_iter_per_epoch[phase]
                loss_per_epoch['kl_ve_gauss'] += kl_ve_gauss.item() / num_iter_per_epoch[phase]


                mse=residual
                mse_per_epoch[phase] += mse / num_iter_per_epoch[phase]

                if (ii + 1) % args.print_freq == 0:
                    print(f"[Epoch:{epoch + 1:0>4d}/{args.epochs:0>4d}] {phase}: {ii + 1:0>5d}/{num_iter_per_epoch[phase]:0>5d}, "
                          f"loss={losses.item():.4e}, KLvh_gauss={kl_vh_gauss.item():.4e}, KLve_gauss={kl_ve_gauss.item():.4e}, "
                          f"MSE={mse:.4e}, GradNormC:{args.clip_grad_C:.2e}/{total_norm_C:.2e}, "
                          f"GradNormM:{args.clip_grad_M:.2e}/{total_norm_M:.2e}, "
                          f"lrD={lr_C:.1e}, lrK={lr_M:.1e}")

                    writer.add_scalar('Iter Train Loss', losses.item(), step)
                    # writer.add_scalar('Iter Train NegLogLikelihood', neg_lh.item(), step)
                    writer.add_scalar('Iter Train KLvh_gauss', kl_vh_gauss.item(), step)
                    writer.add_scalar('Iter Train KLve_gauss', kl_ve_gauss.item(), step)
                    writer.add_scalar('Iter Train GradientNorm CNet', total_norm_C, step)
                    writer.add_scalar('Iter Train GradientNorm MNet', total_norm_M, step)
                    writer.add_scalar('Iter Train MSE', mse, step)


                    step += 1

            print(f"{phase}: Epoch:{epoch + 1}, Loss={loss_per_epoch['Loss']:.4e},"
                  f"KLG={loss_per_epoch['kl_vh_gauss']:.4e}, KLDir={loss_per_epoch['kl_ve_gauss']:.4e}, "
                  f"MSE={mse_per_epoch[phase]:.4e}")

            writer.add_scalar('Epoch Train Loss', loss_per_epoch['Loss'], epoch)
            writer.add_scalar('Epoch Train KL vh_gauss', loss_per_epoch['kl_vh_gauss'], epoch)
            writer.add_scalar('Epoch Train KL ve_gauss', loss_per_epoch['kl_ve_gauss'], epoch)
            # writer.add_scalar('Epoch Train NegLH', loss_per_epoch['NegLH'], epoch)
            writer.add_scalar('Epoch Train MSE', mse_per_epoch[phase], epoch)
            writer.add_scalar('Epoch Train Grad Norm CNet', grad_norm_C, epoch)
            writer.add_scalar('Epoch Train Grad Norm MNet', grad_norm_M, epoch)
            writer.add_scalar('Learning rate CNet', lr_C, epoch)
            writer.add_scalar('Learning rate MNet', lr_M, epoch)
        # torch.cuda.empty_cache()
        print('-' * 150)
        if (epoch + 1) >= args.epoch_start_test or ((epoch + 1) % 500 == 0):
            cnet.eval()
            mnet.eval()

            psnr_per_iter = {x: [] for x in _modes[1:]}

            mse_per_iter = {x: [] for x in _modes[1:]}

            for phase in _modes[1:]:
                for ii, data in enumerate(data_loaders[phase]):


                    y, mR = data[0].to(device), data[1].to(device)
                    valid_loss = AverageMeter()
                    with torch.no_grad():
                        out_MNet1,out_MNet2 = mnet(y)
                        out_CNet = cnet(y)

                        net_output = get_od_mearsurement_sample(out_CNet, out_MNet1, out_MNet2, args.batch_size,
                                                          args.patch_size)
                        valid_residual = criterion(net_output, y)


                        loss_valid,kl_vh_gauss_valid, kl_ve_gauss_valid = loss_fn(out_MNet1,out_MNet2,
                                                            args.alpha_h2,args.alpha_e2, mR
                                                                         )
                        # loss = loss_valid
                        loss = loss_valid * al + (1 - al) * valid_residual
                        valid_loss.update(loss.data)
                        valid_loss_per_epoch['Valid_Loss'] += loss.item() / num_iter_per_epoch[phase]
                        # valid_loss_per_epoch['Valid_NegLH'] += neg_lh_valid.item() / num_iter_per_epoch[phase]
                        valid_loss_per_epoch['Valid_kl_vh_gauss'] += kl_vh_gauss_valid.item() / num_iter_per_epoch[phase]
                        valid_loss_per_epoch['Valid_kl_ve_gauss'] += kl_ve_gauss_valid.item() / num_iter_per_epoch[phase]

                    mse=valid_residual
                    mse_per_epoch[phase] += mse
                    mse_per_iter[phase].append(mse)

                    writer.add_scalar('Iter Valid Loss', loss.item(), step)
                    # writer.add_scalar('Iter Valid NegLogLikelihood',  neg_lh_valid.item(), step)
                    writer.add_scalar('Iter Valid KLvh_gauss', kl_vh_gauss_valid.item(), step)
                    writer.add_scalar('Iter Valid KLve_gauss', kl_ve_gauss_valid.item(), step)



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
                'deblur_model_state_dict': cnet.state_dict(),
                'kernel_model_state_dict': mnet.state_dict(),
                'optimizer_c_state_dict': optimizer_c.state_dict(),
                'optimizer_m_state_dict': optimizer_m.state_dict(),
            }, save_path_model)
            save_path_concentration_model_state = os.path.join(args.model_dir, 'model_state_c_' + str(epoch + 1))
            torch.save(cnet.state_dict(), save_path_concentration_model_state)
            save_path_color_model_state = os.path.join(args.model_dir, 'model_state_m_' + str(epoch + 1))
            torch.save(mnet.state_dict(), save_path_color_model_state)
        toc = time.time()
        print(f'This epoch take time {toc - tic:.2f} s.')
    writer.close()
    print('Reach the maximal epochs! Finish training')


if __name__ == '__main__':
    main()
