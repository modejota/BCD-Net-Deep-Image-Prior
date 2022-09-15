import time
import os
import sys
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np



M=torch.tensor([[0.6442, 0.0928],
         [0.7166, 0.9541],
         [0.2668, 0.2831]])

M=M.T
print(M)
print(M.shape)

out_KNet1=M.repeat(16,1,1)
print(out_KNet1.shape)

var=torch.tensor([0.01, 0.9])
out_KNet2=var.repeat(16,1)
print(out_KNet2.shape)


p=16
V=torch.zeros(p,2,3)


rad=torch.normal(mean=torch.zeros([p,2,3]),std=1)##########16,2,3


V[:, :, 0] = out_KNet2
V[:, :, 1] = out_KNet2
V[:, :, 2] = out_KNet2

v = V * rad  ####16,2,3


M = out_KNet1+v#########16,2,3
M=M.clamp(0,None)

l = torch.norm(M, dim=2, keepdim=True)
l_ = 1.0 / (l + 1e-10)
M = M * l_
print(M.shape)
def od2rgb(od):
    rgb=256*np.exp(-od)-1
    return rgb

def rgb2od(rgb):
    od=(-np.log((rgb+1)/256))
    return od



def display_M(M):
    ns=M.shape[1]
    Mdisp=M.T.reshape(1,ns,3)
    Mdisp=od2rgb(Mdisp).astype('int')
    plt.imshow(Mdisp)



import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for i in range(p):
#     plt.figure()
    plt.subplot(p,1,i+1)
    Mplot=M[i,:,:].T
    print(Mplot)
    display_M(Mplot.numpy())
    plt.axis('off')
plt.show()

# M = torch.normal(mean=out_KNet1, std=out_KNet2.view(16,2,1).repeat(1,1,3)).clamp(0,None)
#
#
# l = torch.norm(M, dim=2, keepdim=True)
# l_ = 1.0 / (l + 1e-10)
# M = M * l_
#
# plt.figure(figsize=(10,10))
# for i in range(p):
# #     plt.figure()
#     plt.subplot(p,1,i+1)
#     Mplot=M[i,:,:].T
#     print(Mplot)
#     display_M(Mplot.numpy())
#     plt.axis('off')
# plt.show()
