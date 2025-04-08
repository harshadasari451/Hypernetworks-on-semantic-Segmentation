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
        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size * self.in_size * self.f_size * self.f_size)), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size * self.in_size * self.f_size * self.f_size)), 2))

        self.w_img = Parameter(torch.fmod(torch.randn((267, 16)), 2))
        self.b_img = Parameter(torch.fmod(torch.randn((16)), 2))

        self.w_pixel = Parameter(torch.fmod(torch.randn((16, 2)), 2))
        
        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, 2 * self.in_size)), 2))
        self.b2 = Parameter(torch.fmod(torch.randn(2 * self.in_size), 2))

    def forward(self, z, y_mask_img, key_pixel):
        # Ensure inputs are on the correct device
        z = z
        y_mask_img = y_mask_img
        key_pixel = key_pixel

        h_z = torch.matmul(z, self.w2) + self.b2
        
        y_mask_img = torch.squeeze(y_mask_img, dim=0).float().flatten()
        y_prime = torch.matmul(y_mask_img, self.w_img) + self.b_img
        
        y_pixel = key_pixel.view(2, -1).float()
        y_Pixel_w = torch.matmul(self.w_pixel, y_pixel).squeeze()
        
        h_img_trans = torch.cat([y_prime, y_Pixel_w])
        h_prime = torch.cat([h_z, h_img_trans])
        h_final = torch.matmul(h_prime.unsqueeze(0), self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel