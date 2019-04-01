import torch
import torch.nn as nn

def conv_inst_lrelu(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size=(3,3), stride=2, bias=False, padding=1),
        nn.InstanceNorm2d(out_chan),
        nn.LeakyReLU(inplace=True)
    )

class SPADEEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer1 = conv_inst_lrelu(3, 64)
        self.layer2 = conv_inst_lrelu(64, 128)
        self.layer3 = conv_inst_lrelu(128, 256)
        self.layer4 = conv_inst_lrelu(256, 512)
        self.layer5 = conv_inst_lrelu(512, 512)
        self.layer6 = conv_inst_lrelu(512, 512)
        self.linear_mean = nn.Linear(8192, args.gen_input_size)
        self.linear_var = nn.Linear(8192, args.gen_input_size)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)

        return self.linear_mean(x), self.linear_var(x)
