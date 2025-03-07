import numpy as np

def get_positional_embedding(x, y, periods=[10, 5]):
    """
    Compute positional embedding for a given (x, y) coordinate.

    Args:
        x (int or float): x-coordinate.
        y (int or float): y-coordinate.
        periods (list): List of periods for sine and cosine functions.

    Returns:
        np.ndarray: Positional embedding as a 1D array.
    """
    freqs = [2 * np.pi / p for p in periods]

    embeddings = []
    for freq in freqs:
        embeddings.extend([np.sin(freq * x), np.cos(freq * x),  # x embeddings
                          np.sin(freq * y), np.cos(freq * y)]) # y embeddings
    return np.array(embeddings)