import torch
from torch.nn import Module, Conv2d
from torch.nn.functional import relu
from torch.nn.utils import spectral_norm
from spade import SPDAE

class SPADEResBlk(Module):
    def __init__(self, args, k):
        super().__init__()
        kernel_size = args.spade_resblk_kernel
        self.spade1 = SPADE(k)
        self.conv1 = Conv2d(k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
        self.spade2 = SPADE(k)
        self.conv2 = Conv2d(k, k, kernel_size=(kernel_size, kernel_size), padding=1, bias=False)
    
    def forward(self, x, seg):
        x_skip = x
        x = relu(self.spade1(x, seg))
        x = self.conv1(x)
        x = relu(self.spade2(x, seg))
        x = self.conv2(x)

        return x_skip + x