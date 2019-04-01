import torch
from torch.nn import Module, Linear, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import tanh, interpolate 
from .spade_resblk import SPADEResBlk

class SPADEGenerator(Module):
    def __init__(self, args):
        super().__init__()
        self.linear = Linear(args.gen_input_size, args.gen_hidden_size)
        self.spade_resblk1 = SPADEResBlk(args, 1024)
        self.spade_resblk2 = SPADEResBlk(args, 1024)
        self.spade_resblk3 = SPADEResBlk(args, 1024)
        self.spade_resblk4 = SPADEResBlk(args, 512)
        self.spade_resblk5 = SPADEResBlk(args, 256)
        self.spade_resblk6 = SPADEResBlk(args, 128)
        self.spade_resblk7 = SPADEResBlk(args, 64)
        self.conv = spectral_norm(Conv2d(64, 3, kernel_size=(3,3), padding=1))

    def forward(self, x, seg):
        b, c, h, w = seg.size()
        x = self.linear(x)
        x = x.view(b, -1, 4, 4)

        x = interpolate(self.spade_resblk1(x, seg), size=(2*h, 2*w), mode='nearest')
        x = interpolate(self.spade_resblk2(x, seg), size=(4*h, 4*w), mode='nearest')
        x = interpolate(self.spade_resblk3(x, seg), size=(8*h, 8*w), mode='nearest')
        x = interpolate(self.spade_resblk4(x, seg), size=(16*h, 16*w), mode='nearest')
        x = interpolate(self.spade_resblk5(x, seg), size=(32*h, 32*w), mode='nearest')
        x = interpolate(self.spade_resblk6(x, seg), size=(64*h, 64*w), mode='nearest')
        x = interpolate(self.spade_resblk7(x, seg), size=(128*h, 128*w), mode='nearest')
        
        x = tanh(self.conv(x))

        return x