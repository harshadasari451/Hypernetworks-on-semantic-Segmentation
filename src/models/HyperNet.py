import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class HyperNetwork(nn.Module):
    def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        # Define parameters and move them to GPU
        self.w1 = Parameter(torch.fmod(torch.randn((16, self.out_size * self.in_size * self.f_size * self.f_size)), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size * self.in_size * self.f_size * self.f_size)), 2))

        self.w_global_img = Parameter(torch.fmod(torch.randn((371, 64)), 2))
        self.b_global_img = Parameter(torch.fmod(torch.randn((64)), 2))

        self.w_expert_img = Parameter(torch.fmod(torch.randn((251, 64)), 2))
        self.b_expert_img = Parameter(torch.fmod(torch.randn((64)), 2))

        self.dense_w = Parameter(torch.fmod(torch.randn((144, 16)), 2))
        self.dense_b = Parameter(torch.fmod(torch.randn((16)), 2))
        
        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size)), 2))
        self.b2 = Parameter(torch.fmod(torch.randn(self.in_size), 2))

    def forward(self, z, expert_patch, global_patch):

        h_z = torch.matmul(z, self.w2) + self.b2
        
        # y_global_img = torch.squeeze(global_patch, dim=0).float()
        y_global_prime = torch.matmul(global_patch, self.w_global_img) + self.b_global_img
        # y_global_prime = torch.tanh(y_global_prime)

        
        # y_expert_img = torch.squeeze(expert_patch, dim=0).float()
        y_expert_prime = torch.matmul(expert_patch, self.w_expert_img) + self.b_expert_img
        # y_expert_prime = torch.tanh(y_expert_prime)

        h_img_trans = torch.cat([y_global_prime, y_expert_prime, h_z], dim = 0)

        h_prime = torch.matmul(h_img_trans, self.dense_w) + self.dense_b  
    
        h_final = torch.matmul(h_prime, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel