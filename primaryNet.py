import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from HyperNet import HyperNetwork
from Resnet import ResNetBlock


class Embedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h,k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2)))

    def forward(self, hyper_net, y_mask_img, key_pixel):
        ww = []
        h, k = self.z_num
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j],y_mask_img, key_pixel))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # # Adding a conv2d for y_mask_img
        # self.y_conv = nn.Conv2d(1,16,3,padding = 1) 
        # self.y_bn = nn.BatchNorm2d(16)

        self.z_dim = z_dim
        self.hope = HyperNetwork(z_dim=self.z_dim)

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[16,16], [16,16], [16,16], [16,16], [16,16], [16,16], [16,32], [32,32], [32,32], [32,32],
                            [32,32], [32,32], [32,64], [64,64], [64,64], [64,64], [64,64], [64,64]]

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResNetBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = nn.ModuleList()

        for i in range(36):
            self.zs.append(Embedding(self.zs_size[i], self.z_dim))

        self.global_avg = nn.AvgPool2d(9)
        self.final = nn.Linear(64,20) # i need to check for number of class in cityscapes file 

    def forward(self, x , y_mask_img, key_pixel):
        
        x = x
        x = F.relu(self.bn1(self.conv1(x)))
        # y_mask_img = y_mask_img.unsqueeze(0)  
        # y_mask = self.y_bn(self.y_conv(y_mask_img.float()))
        # print(y_mask.shape)
        for i in range(18):
            # if i != 15 and i != 17:
            # print(i)
            w1 = self.zs[2*i](self.hope, y_mask_img, key_pixel)
            # print(f"w1 shape: {w1.shape}")
            w2 = self.zs[2*i+1](self.hope, y_mask_img, key_pixel)
            # print(f"w2 shape: {w2.shape}")
            x = self.res_net[i](x, w1, w2)
            # print(f"x shape: {x.shape}")

        x = self.global_avg(x)
        x= x.squeeze(-1).squeeze(-1)
        x = self.final(x)
        # print(x.shape)

        return x