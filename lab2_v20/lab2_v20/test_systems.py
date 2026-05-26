import numpy as np


def get_system():
    A = np.array([
        [4.0, 0.24, -0.08],
        [0.09, 3.0, -0.15],
        [0.04, -0.08, 4.0]
    ], dtype=float)

    b = np.array([8.0, 9.0, 20.0], dtype=float)

    return A, b