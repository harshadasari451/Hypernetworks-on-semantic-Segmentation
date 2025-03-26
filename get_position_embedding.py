import torch
import numpy as np

def get_positional_embedding(x, y, periods=[9, 4]):
    """
    Compute positional embedding for a given (x, y) coordinate.

    Args:
        x (int or float): x-coordinate.
        y (int or float): y-coordinate.
        periods (list of int or float): List of periods for sine and cosine functions.

    Returns:
        torch.Tensor: Positional embedding as a 1D tensor.
    """
    freqs = [2 * np.pi / p for p in periods]

    embeddings = []
    for freq in freqs:
        embeddings.extend([
            np.sin(freq * x), np.cos(freq * x),  # x embeddings
            np.sin(freq * y), np.cos(freq * y)   # y embeddings
        ])
    
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    return embeddings