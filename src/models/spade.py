import torch
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate, relu

import warnings
warnings.filterwarnings("ignore")

class SPADE(Module):
    def __init__(self, args, k):
        super().__init__()
        num_filters = args.spade_filter
        kernel_size = args.spade_kernel
        self.conv = spectral_norm(Conv2d(1, num_filters, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_gamma = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_beta = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))

    def forward(self, x, seg):
        N, C, H, W = x.size()

        sum_channel = torch.sum(x.reshape(N, C, H*W), dim=-1)
        mean = sum_channel / (N*H*W)
        std = torch.sqrt((sum_channel**2 - mean**2) / (N*H*W))

        mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
        std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        x = (x - mean) / std

        seg = interpolate(seg, size=(H,W), mode='nearest')
        seg = relu(self.conv(seg))

        x = torch.matmul(self.conv_gamme(seg), x) + self.conv_beta(seg)

        return x