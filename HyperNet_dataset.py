import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class CityscapesSegmentation(Dataset):
    def __init__(self, image_dir, label_dir, mask_function, get_boundaries, get_patch, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))
        self.mask_function = mask_function
        self.get_boundaries = get_boundaries
        self.get_patch = get_patch
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image = np.array(image)
        label = np.array(label)

        # Generate x2 (masked image)
        masked_image, key_pixels = self.mask_function(label)

        x1_patches = []
        y_labels = []

        # Extract boundary pixels and patches
        for pixel in key_pixels:  # Loop through selected key pixels (p1, p2, p3)
            patches = []
            boundary_pixels = self.get_boundaries(pixel, label)

            for boundary_pixel in boundary_pixels:
                patch = self.get_patch(boundary_pixel, image)
                patches.append(patch)
                y_labels.append(label[boundary_pixel[1], boundary_pixel[0]])  # Get class label
            
            x1_patches.append(patches)


        x2 = torch.tensor(masked_image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W)
        x1_patches = torch.tensor(np.array(x1_patches), dtype=torch.float32).permute(0, 3, 1, 2)  # (N, C, H, W)
        y_labels = torch.tensor(y_labels, dtype=torch.long)

        return x1_patches, x2, y_labels
