def get_boundary_pixels(x, y, image_size, patch_size):
    """
    Get the boundary pixels around a patch centered at (x, y).

    Args:
        x (int): The x-coordinate of the center pixel.
        y (int): The y-coordinate of the center pixel.
        image_size (tuple): The size of the image as (width, height).
        patch_size (int): The size of the patch

    Returns:
        list: A list of (x, y) tuples representing the boundary pixels.
    """
    width, height = image_size
    boundary_pixels = []

    # Define the patch size 
    patch_size = patch_size
    half_patch = patch_size // 2

    # Define the boundary limits (1 pixel outside the patch)
    min_x = x - half_patch - 1
    max_x = x + half_patch + 1
    min_y = y - half_patch - 1
    max_y = y + half_patch + 1

    # Iterate over the boundary pixels
    for i in range(min_x, max_x + 1):
        for j in range(min_y, max_y + 1):
            # Skip pixels inside the patch
            if (i >= x - half_patch and i <= x + half_patch) and (j >= y - half_patch and j <= y + half_patch):
                continue
            # Include only valid pixels within the image boundaries
            if 0 <= i < width and 0 <= j < height:
                boundary_pixels.append((i, j))

    return boundary_pixels


# # Example image size
# image_size = (256, 256)

# # Example pixel location
# x, y = 3, 2
# patch_size = 5
# # Get boundary pixels
# boundary_pixels = get_boundary_pixels(x, y, image_size, patch_size)
# print(boundary_pixels)