import numpy as np


def gauss_solve(A, b):
    n = len(A)

    for k in range(n):
        max_row = k
        for i in range(k + 1, n):
            if abs(A[i, k]) > abs(A[max_row, k]):
                max_row = i

        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]

    return x