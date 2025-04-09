import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
from pathlib import Path

from src.utils.hyp_input import hyp_input
from src.utils.get_boundary_pixels import get_boundary_pixels
from src.utils.extract_patch import extract_patch


class Battery_unet_hyp_data(Dataset):
    def __init__(self, image_dir,unet_model, device, mask_function=hyp_input, get_boundaries=get_boundary_pixels, get_patch=extract_patch, transform=None):
        self.image_dir = image_dir
        # self.label_dir = label_dir
        self.mask_function = mask_function
        self.get_boundaries = get_boundaries
        self.get_patch = get_patch
        self.transform = transform
        self.unet_model = unet_model
        self.device = device
        
        self.image_files = sorted(Path(image_dir).glob('*.png'))
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]

        # Load and transform images
        image = Image.open(image_path).convert('L')
        img_ndarray = np.asarray(image)
        img_ndarray = img_ndarray[np.newaxis, ...]  # Add channel dimension [1, H, W]
        image_tensor = torch.as_tensor(img_ndarray / 255.0).float().contiguous()
        
        with torch.no_grad():
            input_img = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            label_tensor = self.unet_model(input_img)
            label_tensor = torch.argmax(label_tensor, dim = 1)
            print()
        label_tensor = label_tensor.squeeze(0).cpu().type(torch.long)
        

        _, H, W = image_tensor.shape

        # Get key pixels and masked image
        key_pixels, masked_image = self.mask_function(label_tensor)

        all_patches = []
        all_labels = []

        mismatch = 0

        for x, y in key_pixels:
            boundary_pixels = self.get_boundaries(x, y)

            patches = []
            labels = []

            for bx, by in boundary_pixels:
                if bx < 0 or by < 0 or bx >= H or by >= W:
                    patches.append(torch.zeros((1, 9, 9), dtype=torch.long))
                    labels.append(255)
                    mismatch += 1
                else:
                    patches.append(self.get_patch(image_tensor, bx, by))
                    labels.append(label_tensor[bx, by])  # Get label ID
            
            all_patches.append(torch.stack(patches))  # Shape: (max_boundaries, C, H, W)
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        # Convert lists to tensors
        all_patches = torch.stack(all_patches)  # Shape: (num_key_pixels, max_boundaries, C, H, W)
        all_labels = torch.stack(all_labels)  # Shape: (num_key_pixels, max_boundaries)

        return all_patches, masked_image, key_pixels, all_labels, mismatch
