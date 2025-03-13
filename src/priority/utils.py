import numpy as np


def U(size):
    return np.random.uniform(size=size)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.sum(x, dtype=float)
