import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import glob
from torchvision import transforms
from cityscapesscripts.helpers.labels import id2label

from Hypernetworks_stevens import extract_patch
from Hypernetworks_stevens import get_boundary_pixels
from Hypernetworks_stevens import hyp_input



class CityscapesSegmentation(Dataset):
    def __init__(self, image_dir, label_dir, mask_function=hyp_input, get_boundaries=get_boundary_pixels, get_patch=extract_patch, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mask_function = mask_function
        self.get_boundaries = get_boundaries
        self.get_patch = get_patch
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(image_dir, "**", "*_leftImg8bit.png"), recursive=True))
        self.label_files = sorted(glob.glob(os.path.join(label_dir, "**", "*_gtFine_labelIds.png"), recursive=True))

        assert len(self.image_files) == len(self.label_files), "Number of image and label files must be the same!"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts image to a tensor (C, H, W)
        ])

        # Load and transform images
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)

        image = transform(image)
        label = torch.tensor(np.array(label), dtype=torch.long)  # Convert label to tensor (H, W)

        _, H, W = image.shape

        # Map label IDs to train IDs safely
        label_id_image = torch.tensor(
            np.vectorize(lambda x: id2label[x].trainId if x in id2label else 255)(label.numpy()),
            dtype=torch.long
        )

        # Get key pixels and masked image
        key_pixels, masked_image = self.mask_function(label_id_image)

        all_patches = []
        all_labels = []

        for x, y in key_pixels:
            boundary_pixels = self.get_boundaries(x, y, H, W)

            patches = []
            labels = []

            for bx, by in boundary_pixels:
                if bx < 0 or by < 0 or bx >= H or by >= W:
                    patches.append(torch.zeros((3, 9, 9)))
                    labels.append(255)
                else:
                    patches.append(self.get_patch(image, bx, by))
                    labels.append(label_id_image[bx, by])  # Get label ID

            

            all_patches.append(torch.stack(patches))  # Shape: (max_boundaries, C, H, W)
            all_labels.append(torch.tensor(labels, dtype=torch.long))

        # Convert lists to tensors
        all_patches = torch.stack(all_patches)  # Shape: (num_key_pixels, max_boundaries, C, H, W)
        all_labels = torch.stack(all_labels)  # Shape: (num_key_pixels, max_boundaries)

        return all_patches, masked_image, key_pixels, all_labels
