import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch.nn.utils import spectral_norm

def custom_model1(in_chan, out_chan):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4,4), stride=2, padding=1)),
        nn.LeakyReLU(inplace=True)
    )

def custom_model2(in_chan, out_chan, stride=2):
    return nn.Sequential(
        spectral_norm(nn.Conv2d(in_chan, out_chan, kernel_size=(4,4), stride=stride, padding=1)),
        nn.InstanceNorm2d(out_chan),
        nn.LeakyReLU(inplace=True)
    )

class SPADEDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer1 = custom_model1(4, 64)
        self.layer2 = custom_model2(64, 128)
        self.layer3 = custom_model2(128, 256)
        self.layer4 = custom_model2(256, 512, stride=1)
        self.inst_norm = nn.InstanceNorm2d(512)
        self.conv = spectral_norm(nn.Conv2d(512, 1, kernel_size=(4,4), padding=1))

    def forward(self, img, seg):
        x = torch.cat((seg, img.detach()), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = leaky_relu(self.inst_norm(x))
        x = self.conv(x)
        return x
        