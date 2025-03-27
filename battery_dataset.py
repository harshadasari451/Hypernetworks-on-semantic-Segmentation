import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
from pathlib import Path

from Hypernetworks_stevens import get_boundary_pixels, extract_patch, hyp_input




class Battery_data(Dataset):
    def __init__(self, image_dir, label_dir, mask_function=hyp_input, get_boundaries=get_boundary_pixels, get_patch=extract_patch, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_function = mask_function
        self.get_boundaries = get_boundaries
        self.get_patch = get_patch
        self.transform = transform
        


        self.image_files = sorted(Path(image_dir).glob('*.png'))
        self.label_files = sorted(Path(label_dir).glob('*.png'))

        assert len(self.image_files) == len(self.label_files), "Number of image and label files must be the same!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # image_transform = transforms.Compose([
        #     transforms.Lambda(lambda x: x.convert('L') if x.mode != 'L' else x),
        #     transforms.Lambda(lambda x: np.asarray(x)[np.newaxis, ...] / 255.0),
        #     transforms.ToTensor(),
        # ])

        mask_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('L') if x.mode != 'L' else x),
            transforms.Lambda(lambda x: np.array(x, dtype=np.float32) / 255.0),
            transforms.Lambda(lambda x: np.where((x > 0) & (x < 1.0), 2.0, x)),
            transforms.Lambda(lambda x: torch.as_tensor(x.copy()).long()),
        ])

        # Load and transform images
        image = Image.open(image_path).convert('L')
        label = Image.open(label_path)
        img_ndarray = np.asarray(image)
        img_ndarray = img_ndarray[np.newaxis, ...]  # Add channel dimension [1, H, W]
        image_tensor = torch.as_tensor(img_ndarray / 255.0).float().contiguous()
        
        label_tensor = mask_transform(label)  
        

        _, H, W = image_tensor.shape

        # Get key pixels and masked image
        key_pixels, masked_image = self.mask_function(label_tensor)

        all_patches = []
        all_labels = []

        # mismatch = 0

        for x, y in key_pixels:
            boundary_pixels = self.get_boundaries(x, y)

            patches = []
            labels = []

            for bx, by in boundary_pixels:
                if bx < 0 or by < 0 or bx >= H or by >= W:
                    patches.append(torch.zeros((1, 9, 9)))
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

        return all_patches, masked_image, key_pixels, all_labels #, mismatch
