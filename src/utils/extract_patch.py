import torch


def extract_patch(image, x, y, patch_size=(9, 9)):
    """
    Extract a patch centered at (x, y) from an image tensor.
    If the patch goes out of bounds, pad with zeros.
    
    Args:
        image (torch.Tensor): Input image tensor with shape (C, H, W).
        x (int): X-coordinate of the center pixel.
        y (int): Y-coordinate of the center pixel.
        patch_size (tuple): Size of the patch (height, width).
    
    Returns:
        torch.Tensor: Extracted patch with shape (C, patch_height, patch_width).

    """
    if image.ndim != 3:
        image = image.unsqueeze(0)  # Add batch dimension

    C, H, W = image.shape  # Ensure correct channel, height, width ordering
    patch_height, patch_width = patch_size

    # Calculate the region of the image to extract
    x_start = x - patch_height // 2
    x_end = x_start + patch_height
    y_start = y - patch_width // 2
    y_end = y_start + patch_width

    # Initialize the patch with zeros
    patch = torch.zeros((C, patch_height, patch_width), dtype=torch.float32)

    # Compute valid range within image
    valid_x_start = max(x_start, 0)
    valid_x_end = min(x_end, H)
    valid_y_start = max(y_start, 0)
    valid_y_end = min(y_end, W)

    # Compute where to place the extracted image pixels in the patch
    patch_x_start = max(0, -x_start)  # Offset if out of bounds on the left
    patch_x_end = patch_x_start + (valid_x_end - valid_x_start)
    patch_y_start = max(0, -y_start)  # Offset if out of bounds on the top
    patch_y_end = patch_y_start + (valid_y_end - valid_y_start)

    # Copy the valid region from the image into the patch
    patch[:, patch_x_start:patch_x_end, patch_y_start:patch_y_end] = \
        image[:, valid_x_start:valid_x_end, valid_y_start:valid_y_end]

    return patch
