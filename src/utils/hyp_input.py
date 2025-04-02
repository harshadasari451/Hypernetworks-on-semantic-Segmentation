import torch

from Hypernetworks_stevens import extract_patch
from Hypernetworks_stevens import get_positional_embedding

def hyp_input(img_tensor, patch_size=(9,9), num_patches=3):
    """
    Extracts random patches from img_tensor and computes positional embeddings for the patches.

    Args:
        img_tensor (torch.Tensor): The input image tensor of shape (C, H, W)
        patch_size (int): The size of the extracted patches
        num_patches (int): The number of patches to extract (default is 3)

    Returns:
        torch.Tensor: The indices of the selected pixels (num_patches, 2)
        torch.Tensor: The tensor containing flattened patches concatenated with positional embeddings
    """
    if img_tensor.ndim != 3:
        img_tensor = img_tensor.unsqueeze(0)

    _ ,H, W = img_tensor.shape  # Assuming shape ( H, W)

    # Randomly select pixel locations (y, x) ensuring patches fit within bounds
    x_coords = torch.randint(0, H , (num_patches,))
    y_coords = torch.randint(0, W , (num_patches,))
    xy_coords = torch.stack([x_coords, y_coords], dim=1)  # Shape: (num_patches, 2)

    patches = []

    for x,y in xy_coords:
        patch = extract_patch(img_tensor, x.item(), y.item(), (patch_size))

        position_embedding = get_positional_embedding(x.item(), y.item())

        patch_tensor = torch.cat([patch.flatten(), position_embedding])
        patches.append(patch_tensor)

    final_tensor = torch.stack(patches)  # Shape: (num_patches, flattened_patch + positional_embedding)

    return xy_coords, final_tensor