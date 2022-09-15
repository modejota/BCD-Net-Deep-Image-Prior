import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels, out_channels, bias=True):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv5x5(in_channels, out_channels, bias=True):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=bias)
    return layer


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, strides=1, slope=0.2):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=kernel_size // 2),
            nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=kernel_size // 2),
        ])

    def forward(self, x):
        out = self.layers(x) + x
        return out




