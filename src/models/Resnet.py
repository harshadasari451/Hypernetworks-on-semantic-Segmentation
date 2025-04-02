import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityLayer(nn.Module):
    def forward(self, x):
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_size=16, out_size=16, downsample=False):
        super(ResNetBlock, self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=1, kernel_size=3, padding=1) # some mistake. need to fix it
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x, conv1_w, conv2_w):
        x = x
        residual = self.reslayer(x)
        # print(f"Residual shape: {residual.shape}")

        out = F.relu(self.bn1(F.conv2d(x, conv1_w, stride=1, padding=1)), inplace=True)
        # print(f"Conv1 shape: {out.shape}")
        out = self.bn2(F.conv2d(out, conv2_w, padding=1))

        out += residual
        # print(f"Out shape: {out.shape}")

        out = F.relu(out)

        return out
