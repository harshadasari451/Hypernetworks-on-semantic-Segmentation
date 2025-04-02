def get_boundary_pixels(x, y,patch_size=(9,9)):
    x_half_patch = patch_size[0] // 2
    y_half_patch = patch_size[1] // 2

    # Allow negative values and values beyond image dimensions
    x_min = x - x_half_patch - 1
    x_max = x + x_half_patch + 1
    y_min = y - y_half_patch - 1
    y_max = y + y_half_patch + 1

    boundary_pixels = []

    # Top and bottom boundaries (including corners)
    for i in range(x_min, x_max + 1):
        boundary_pixels.append((i, y_min))  # Top boundary
        boundary_pixels.append((i, y_max))  # Bottom boundary

    # Left and right boundaries (excluding corners to avoid duplicates)
    for j in range(y_min + 1, y_max):
        boundary_pixels.append((x_min, j))  # Left boundary
        boundary_pixels.append((x_max, j))  # Right boundary

    return boundary_pixels