import time
import os
import sys
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.utils_metric import batch_PSNR, batch_SSIM
from datasets.deblur_main_dataset import get_dataloaders
from loss import loss_fn
from networks.cnet import get_dnet
from networks.mnet import get_knet
from options import set_opts


args = set_opts()
for arg in vars(args):
    print('{:<25s}: {:s}'.format(arg, str(getattr(args, arg))))


modes_all = {"Text": ["train_Text", "test_Text"],
             "GoPro": ["train_GoPro", "test_GoPro"],
             "RealBlur_J":  ["train_RealBlur_J", "test_RealBlur_J"],
             "RealBlur_R":  ["train_RealBlur_R", "test_RealBlur_R"]}


def adjust_learning_rate(optimizer, epoch, args):
    if args.run_mode == "Text":
        if epoch <= 60:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        elif epoch <= 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6

    elif args.run_mode == "GoPro":
        if epoch <= 270:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        elif epoch <= 290:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6

    elif args.run_mode == "RealBlur_J" or args.run_mode == "RealBlur_R":
        if epoch <= 3800:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        elif epoch <= 4000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-6


def main():
    dnet = get_dnet(args.DeblurNet)
    knet = get_knet(args.KNet, kernel_size=args.max_size)
    dnet = dnet.cuda()
    knet = knet.cuda()
    optimizer_d = optim.Adam(dnet.parameters(), args.lr_D)
    optimizer_k = optim.Adam(knet.parameters(), args.lr_K)

    dirichlet_para_stretch = args.dirichlet_para_stretch
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
        loss_per_epoch = {x: 0 for x in ['Loss', 'NegLH', 'KLGaussian', 'KLDirichlet']}
        mse_per_epoch = {x: 0 for x in _modes}
        # 为了让训练更稳定加入一个预训练模糊核的过程
        if epoch < pretraining_epoch:
            pre_l1_loss = 0
            for ii, data in enumerate(data_loaders[_modes[0]]):
                im_gt = data[0].cuda()  # B3HW
                im_blurry = data[1].cuda()
                blur_kernel_gt = data[2].cuda()  # BHW
                blur_kernel_gt = blur_kernel_gt.unsqueeze(1)  # B x 1 x kernel_size x kernel_size
                pre_optimizer_k.zero_grad()
                out_KNet = knet(im_blurry)
                kernel_l1_loss = F.l1_loss(out_KNet, blur_kernel_gt)
                kernel_l1_loss.backward()
                pre_optimizer_k.step()
                pre_l1_loss += kernel_l1_loss.item() / num_iter_per_epoch[_modes[0]]
            print(f"PreTraining: Epoch:{epoch + 1}, L1Loss={pre_l1_loss:.4e}")
        else:
            adjust_learning_rate(optimizer_k, epoch, args)
            adjust_learning_rate(optimizer_d, epoch, args)
            grad_norm_D = grad_norm_K = 0
            lr_D = optimizer_d.param_groups[0]['lr']
            lr_K = optimizer_k.param_groups[0]['lr']
            phase = _modes[0]

            for ii, data in enumerate(data_loaders[phase]):
                im_gt = data[0].cuda()  # B3HW
                im_blurry = data[1].cuda()
                blur_kernel_gt = data[2].cuda() #BHW
                blur_kernel_gt = blur_kernel_gt.unsqueeze(1)  # B x 1 x kernel_size x kernel_size
                optimizer_k.zero_grad()
                optimizer_d.zero_grad()

                out_KNet = knet(im_blurry)
                BS = im_gt.shape[0]
                xi = out_KNet.reshape(BS, -1) * dirichlet_para_stretch + 1e-7  # 得到参数
                q_k = torch.distributions.dirichlet.Dirichlet(xi)
                kernel_sample_return = q_k.rsample()  #重参数采样
                kernel_sample_return = kernel_sample_return.reshape(BS, 1, args.max_size, -1)
                out_DNet = dnet(im_blurry, kernel_sample_return.detach())
                loss, neg_lh, kl_gaussian, kl_Dirichlet = loss_fn(out_DNet, out_KNet, im_blurry, im_gt, blur_kernel_gt,
                                                           kernel_sample_return, args.alpha2, args.beta2,
                                                           dirichlet_para_stretch)
                loss.backward()
                total_norm_D = nn.utils.clip_grad_norm_(dnet.parameters(), args.clip_grad_D)
                total_norm_K = nn.utils.clip_grad_norm_(knet.parameters(), args.clip_grad_K)
                grad_norm_D = grad_norm_D + total_norm_D / num_iter_per_epoch[phase]
                grad_norm_K = grad_norm_K + total_norm_K / num_iter_per_epoch[phase]
                optimizer_d.step()
                optimizer_k.step()

                loss_per_epoch['Loss'] += loss.item() / num_iter_per_epoch[phase]
                loss_per_epoch['NegLH'] += neg_lh.item() / num_iter_per_epoch[phase]
                loss_per_epoch['KLGaussian'] += kl_gaussian.item() / num_iter_per_epoch[phase]
                loss_per_epoch['KLDirichlet'] += kl_Dirichlet.item() / num_iter_per_epoch[phase]

                im_deblur = torch.clamp(out_DNet[:, :3, :, : ].detach().data, 0.0, 1.0)
                mse = F.mse_loss(im_deblur, im_gt)
                mse_per_epoch[phase] += mse / num_iter_per_epoch[phase]

                if (ii + 1) % args.print_freq == 0:
                    print(f"[Epoch:{epoch + 1:0>4d}/{args.epochs:0>4d}] {phase}: {ii + 1:0>5d}/{num_iter_per_epoch[phase]:0>5d}, "
                          f"NegLH={neg_lh.item():.4e}, KLG={kl_gaussian.item():.4e}, KLDir={kl_Dirichlet.item():.4e}, "
                          f"MSE={mse:.4e}, GradNormD:{args.clip_grad_D:.2e}/{total_norm_D:.2e}, "
                          f"GradNormK:{args.clip_grad_K:.2e}/{total_norm_K:.2e}, "
                          f"lrD={lr_D:.1e}, lrK={lr_K:.1e}")

                    writer.add_scalar('Iter Train Loss', loss.item(), step)
                    writer.add_scalar('Iter Train NegLogLikelihood', neg_lh.item(), step)
                    writer.add_scalar('Iter Train KLGaussian', kl_gaussian.item(), step)
                    writer.add_scalar('Iter Train KLDirichlet', kl_Dirichlet.item(), step)
                    writer.add_scalar('Iter Train GradientNorm DNet', total_norm_D, step)
                    writer.add_scalar('Iter Train GradientNorm KNet', total_norm_K, step)
                    writer.add_scalar('Iter Train MSE', mse, step)

                    x1 = torchvision.utils.make_grid(im_deblur, normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Deblur Image', x1, step)
                    x2 = torchvision.utils.make_grid(im_gt, normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Ground Truth Image', x2, step)
                    x3 = torchvision.utils.make_grid(im_blurry, normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Blurry Image', x3, step)
                    x4 = torchvision.utils.make_grid(out_KNet.detach(), normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Predict Kernel', x4, step)
                    blur_kernel_gt = blur_kernel_gt.reshape(out_KNet.size()[0], 1, args.max_size, -1)
                    x5 = torchvision.utils.make_grid(blur_kernel_gt, normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Ground Truth Kernel', x5, step)
                    step += 1

            print(f"{phase}: Epoch:{epoch + 1}, Loss={loss_per_epoch['Loss']:.4e}, lh={loss_per_epoch['NegLH']:.4e}, "
                  f"KLG={loss_per_epoch['KLGaussian']:.4e}, KLDir={loss_per_epoch['KLDirichlet']:.4e}, "
                  f"MSE={mse_per_epoch[phase]:.4e}")

            writer.add_scalar('Epoch Train Loss', loss_per_epoch['Loss'], epoch)
            writer.add_scalar('Epoch Train KL Gaussian', loss_per_epoch['KLGaussian'], epoch)
            writer.add_scalar('Epoch Train KL Dirichlet', loss_per_epoch['KLDirichlet'], epoch)
            writer.add_scalar('Epoch Train NegLH', loss_per_epoch['NegLH'], epoch)
            writer.add_scalar('Epoch Train MSE', mse_per_epoch[phase], epoch)
            writer.add_scalar('Epoch Train Grad Norm DNet', grad_norm_D, epoch)
            writer.add_scalar('Epoch Train Grad Norm KNet', grad_norm_K, epoch)
            writer.add_scalar('Learning rate DNet', lr_D, epoch)
            writer.add_scalar('Learning rate KNet', lr_K, epoch)

        print('-' * 150)
        if (epoch + 1) >= args.epoch_start_test or ((epoch + 1) % 500 == 0):
            dnet.eval()
            knet.eval()
            psnr_per_epoch = {x: 0 for x in _modes[1:]}
            ssim_per_epoch = {x: 0 for x in _modes[1:]}

            psnr_per_iter = {x: [] for x in _modes[1:]}
            ssim_per_iter = {x: [] for x in _modes[1:]}
            mse_per_iter = {x: [] for x in _modes[1:]}

            for phase in _modes[1:]:
                for ii, data in enumerate(data_loaders[phase]):
                    im_gt = data[0].cuda()  # B3HW
                    im_blurry = data[1].cuda()
                    blur_kernel_gt = data[2].cuda()  # BHW if gopro, kernel=0
                    blur_file_name = data[3]
                    with torch.no_grad():
                        out_KNet = knet(im_blurry)
                        out_DNet = dnet(im_blurry, out_KNet)

                    im_deblur = torch.clamp(out_DNet[:, :3, :, :].detach().data, 0.0, 1.0)
                    mse = F.mse_loss(im_deblur, im_gt)
                    mse_per_epoch[phase] += mse
                    psnr_iter = batch_PSNR(im_deblur, im_gt)
                    ssim_iter = batch_SSIM(im_deblur, im_gt)
                    psnr_per_epoch[phase] += psnr_iter
                    ssim_per_epoch[phase] += ssim_iter

                    psnr_per_iter[phase].append(psnr_iter)
                    ssim_per_iter[phase].append(ssim_iter)
                    mse_per_iter[phase].append(mse)

                    x1 = torchvision.utils.make_grid(im_deblur, normalize=True, scale_each=True)
                    writer.add_image(phase + ' Deblur Image', x1, step_img[phase])
                    x2 = torchvision.utils.make_grid(im_gt, normalize=True, scale_each=True)
                    writer.add_image(phase + ' Ground Truth Image', x2, step_img[phase])
                    x3 = torchvision.utils.make_grid(im_blurry, normalize=True, scale_each=True)
                    writer.add_image(phase + ' Blurry Image', x3, step_img[phase])
                    x4 = torchvision.utils.make_grid(out_KNet.detach(), normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Predict Kernel', x4, step_img[phase])
                    x5 = torchvision.utils.make_grid(blur_kernel_gt, normalize=True, scale_each=True, nrow=args.nrow)
                    writer.add_image(phase + ' Ground Truth Kernel', x5, step_img[phase])
                    step_img[phase] += 1

                psnr_per_epoch[phase] /= (ii + 1)
                ssim_per_epoch[phase] /= (ii + 1)
                mse_per_epoch[phase] /= (ii + 1)

                for k in range(len(psnr_per_iter[phase])):
                    print(f'{phase}, {k + 1:>4d}/{num_iter_per_epoch[phase]:>4d}, mse={mse_per_iter[phase][k]:.4f}, '
                          f'psnr={psnr_per_iter[phase][k]:4.2f}, ssim={ssim_per_iter[phase][k]:5.4f}')
                print(f'{phase:s}: mse={mse_per_epoch[phase]:.3e}, PSNR={psnr_per_epoch[phase]:4.2f}, '
                      f'SSIM={ssim_per_epoch[phase]:5.4f}')
                print('-' * 90)
                writer.add_scalar('Test MSE of ' + phase, mse_per_epoch[phase], epoch)
                writer.add_scalar('Test PSNR of ' + phase, psnr_per_epoch[phase], epoch)
                writer.add_scalar('Test SSIM of ' + phase, ssim_per_epoch[phase], epoch)

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
