import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sum(x, dtype=float)
