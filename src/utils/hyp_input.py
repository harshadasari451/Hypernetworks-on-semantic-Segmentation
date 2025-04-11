import torch

from src.utils.extract_patch import extract_patch
from src.utils.get_position_embedding import get_positional_embedding

def hyp_input(expert_img_tensor, small_img_tensor, patch_size=(9,9), num_patches=3):
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
    if expert_img_tensor.ndim != 3:
        expert_img_tensor = expert_img_tensor.unsqueeze(0)
    if small_img_tensor.ndim != 3:
        small_img_tensor = small_img_tensor.unsqueeze(0)

    _ ,H, W = expert_img_tensor.shape  # Assuming shape ( H, W)

    # Randomly select pixel locations (y, x) ensuring patches fit within bounds
    x_coords = torch.randint(0, H , (num_patches,))
    y_coords = torch.randint(0, W , (num_patches,))
    xy_coords = torch.stack([x_coords, y_coords], dim=1)  # Shape: (num_patches, 2)

    expert_patches = []
    small_patches = []

    for x,y in xy_coords:
        expert_patch = extract_patch(expert_img_tensor, x.item(), y.item(), 9,9)
        small_patch = extract_patch(small_img_tensor, x.item(), y.item(), 11,11)


        expert_position_embedding = get_positional_embedding(x.item(), y.item())
        small_position_embedding = get_positional_embedding(x.item(), y.item(), periods = [11,5])

        expert_patch_tensor = torch.cat([expert_patch.flatten(), expert_position_embedding])
        small_patch_tensor = torch.cat([small_patch.flatten(), small_position_embedding])

        expert_patches.append(expert_patch_tensor)
        small_patches.append(small_patch_tensor)

    expert_final_tensor = torch.stack(expert_patches)  # Shape: (num_patches, flattened_patch + positional_embedding)
    small_final_tensor = torch.stack(small_patches)

    return xy_coords, expert_final_tensor, small_final_tensor