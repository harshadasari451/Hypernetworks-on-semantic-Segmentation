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
    def __init__(self, image_dir,label_dir,expert_model,small_model, device, mask_function=hyp_input, get_boundaries=get_boundary_pixels, get_patch=extract_patch, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_function = mask_function
        self.get_boundaries = get_boundaries
        self.get_patch = get_patch
        self.transform = transform
        self.expert_model = expert_model
        self.small_model = small_model
        self.device = device
        
        self.image_files = sorted(Path(image_dir).glob('*.png'))
        self.label_files = sorted(Path(label_dir).glob('*.png'))
        assert len(self.image_files) == len(self.label_files), "Number of image and label files must be the same!"

        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]
        label = Image.open(label_path)

        mask_transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('L') if x.mode != 'L' else x),
            transforms.Lambda(lambda x: np.array(x, dtype=np.float32) / 255.0),
            transforms.Lambda(lambda x: np.where((x > 0) & (x < 1.0), 2.0, x)),
            transforms.Lambda(lambda x: torch.as_tensor(x.copy()).long()),
        ])
        label_tensor = mask_transform(label) 

        # Load and transform images
        image = Image.open(image_path).convert('L')
        img_ndarray = np.asarray(image)
        img_ndarray = img_ndarray[np.newaxis, ...]  # Add channel dimension [1, H, W]
        image_tensor = torch.as_tensor(img_ndarray / 255.0).float().contiguous()
        
        with torch.no_grad():
            input_img = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
            expert_label_tensor = self.expert_model(input_img)
            small_label_tensor = self.small_model(input_img)
        expert_label_tensor = expert_label_tensor.squeeze(0).cpu().type(torch.long)
        small_label_tensor = small_label_tensor.squeeze(0).cpu().type(torch.long)

        # print(f"expert_label_tensor shape: {expert_label_tensor.shape}")
        # print(f"small_label_tensor shape: {small_label_tensor.shape}")

        

        _, H, W = image_tensor.shape

        # Get key pixels and masked image
        key_pixels, expert_patches, global_patches = self.mask_function(expert_label_tensor, small_label_tensor)
        



        # print(f"masked_img shape: {expert_patches.shape}")
        # print(f"global_patches shape: {global_patches.shape}")

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

        return all_patches, global_patches, expert_patches, all_labels, mismatch
