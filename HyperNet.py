import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class HyperNetwork(nn.Module):
    def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        # Define parameters
        self.w1 = Parameter(torch.fmod(torch.randn((2 * self.z_dim, self.in_size * self.f_size * self.f_size)).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.in_size * self.f_size * self.f_size)).cuda(), 2))

        self.w_img = Parameter(torch.fmod(torch.randn((1024, 64)).cuda(), 2))
        self.b_img = Parameter(torch.fmod(torch.randn((64,)).cuda(), 2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim)).cuda(), 2))

    def forward(self, z, y_mask_img):

        # Compute h_in
        h_in = torch.matmul(z, self.w2) + self.b2  
        h_in = h_in.view(self.in_size, self.z_dim)  

        y_mask_img = torch.squeeze(y_mask_img, dim=0)  # Remove batch dimension
        y_mask_img = y_mask_img.view(16, -1)  # Reshape to (16, 1024)
        h_img_transformed = torch.matmul(y_mask_img, self.w_img) + self.b_img  # Shape: (16, 64)


        h_combined = torch.cat((h_in, h_img_transformed), dim=1) 
        h_final = torch.matmul(h_combined, self.w1) + self.b1  
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size) 
        return kernel