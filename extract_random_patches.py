import torch
import torch.nn.functional as F
import random

def extract_random_patches(img_tensor, patch_size, num_patches = 3):
    """
    Extracts 3 random patches from img_tensor and mask the images excepth the random patchs

    Args:
    - img_tensor (Tensor): The input image tensor of shape (C, H, W)
    - patch_size (int): The size of the extracted patches
    - num_patches (int): The number of patches to extract (default is 3)


    Returns:
    - patches (Tensor): The extracted patches of shape (3, C, patch_size, patch_size)
    - pixel_indices (Tensor): The indices of the selected pixels (3, 2)
    - masked_img (Tensor): The masked image tensor with only the patches visible
    """
    _, H, W = img_tensor.shape  # Assuming shape (C, H, W)
    half_patch = patch_size // 2

    # Randomly select 3 pixel locations (y, x)
    yx_coords = torch.randint(half_patch, H - half_patch, (num_patches, 2))

    patches = []
    masked_img = torch.zeros_like(img_tensor)  # Initialize masked image

    for i, (y, x) in enumerate(yx_coords):
        patch = img_tensor[:, y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
        patches.append(patch)

        # Assign the patch to the masked image
        masked_img[:, y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1] = patch

    patches = torch.stack(patches)  # Shape: (num_patchs, C, patch_size, patch_size)

    return patches, yx_coords, masked_img

# # Example Usage:
# img = torch.rand(1, 128, 128)  # Example image tensor (3 channels, 128x128)
# patches, pixel_indices, masked_img = extract_random_patches(img, patch_size=9)

# print("Extracted Patches Shape:", patches.shape)  # Should be (3, C, 9, 9)
# print("Selected Pixel Indices:", pixel_indices)
# print("Masked Image Shape:", masked_img.shape)  # Should be same as img
