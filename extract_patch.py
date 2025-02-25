import numpy as np

def extract_patch(image, x, y, patch_size=(10, 10)):
    """
    Extract a patch of size patch_size around the pixel (x, y) from the image.
    If the patch goes out of bounds, pad with zeros.

    Args:
        image (numpy.ndarray): Input image (2D or 3D array).
        x (int): X-coordinate of the center pixel.
        y (int): Y-coordinate of the center pixel.
        patch_size (tuple): Size of the patch (height, width).

    Returns:
        numpy.ndarray: Extracted patch.
    """
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size

    # Calculate the start and end indices for the patch
    x_start = x - patch_width // 2
    x_end = x_start + patch_width
    y_start = y - patch_height // 2
    y_end = y_start + patch_height

    patch = np.zeros((patch_height, patch_width) if image.ndim == 2 else (patch_height, patch_width, int(image.shape[2])))

    x_start_img = max(x_start, 0)
    x_end_img = min(x_end, width)
    y_start_img = max(y_start, 0)
    y_end_img = min(y_end, height)

    x_start_patch = x_start_img - x_start
    x_end_patch = x_start_patch + (x_end_img - x_start_img)
    y_start_patch = y_start_img - y_start
    y_end_patch = y_start_patch + (y_end_img - y_start_img)

    patch[y_start_patch:y_end_patch, x_start_patch:x_end_patch] = image[y_start_img:y_end_img, x_start_img:x_end_img]

    return patch

# # Example usage:
# image = np.random.rand(100, 100,3)  
# x, y = 3, 2 
# patch = extract_patch(image, x, y, patch_size=(10, 10))
# print(patch.shape)  # Output: (10, 10,3)