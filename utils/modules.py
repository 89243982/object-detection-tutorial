import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1, leakyReLU=False):
        super(Conv2d, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True) if leakyReLU else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convs(x)

class conv_set(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, iters=1, leakyReLU=False):
        super(conv_set, self).__init__()
        self.iters = iters
        layers = [nn.Sequential(
                    Conv2d(in_channels, inter_channels, 1, leakyReLU=leakyReLU),
                    Conv2d(inter_channels, out_channels, 3, padding=1, leakyReLU=leakyReLU)
                )]

        if iters > 1:
            for _ in range(iters-1):
                layers.append(nn.Sequential(
                    Conv2d(out_channels, inter_channels, 1, leakyReLU=leakyReLU),
                    Conv2d(inter_channels, out_channels, 3, padding=1, leakyReLU=leakyReLU)
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class branch(nn.Module):
    def __init__(self, ch, t=2, leakyReLU=False):
        super(branch, self).__init__()
        self.module_list = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(ch, ch, 1),
            nn.BatchNorm2d(ch)
        )
        self.branch_1 = Conv2d(ch, ch // t, 3, 1, leakyReLU=leakyReLU)
        self.branch_2 = Conv2d(ch, ch // t, 3, padding=2, dilation=2, leakyReLU=leakyReLU)
        self.branch_3 = Conv2d(ch, ch // t, 3, padding=3, dilation=3, leakyReLU=leakyReLU)
        self.fusion = nn.Sequential(
            nn.Conv2d(ch // t * 3, ch, 1),
            nn.BatchNorm2d(ch)
        )

    def forward(self, x):
        x = self.conv1x1(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        x_f = torch.cat([x_1, x_2, x_3], 1)

        return self.relu(self.fusion(x_f))
        
        # return self.relu(x + self.fusion(x_f))
