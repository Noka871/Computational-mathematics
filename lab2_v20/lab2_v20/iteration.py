import numpy as np


def simple_iteration(A, b, eps=0.01, max_iter=100):
    n = len(A)

    B = np.zeros((n, n))
    c = np.zeros(n)

    for i in range(n):
        c[i] = b[i] / A[i, i]
        for j in range(n):
            if i != j:
                B[i, j] = -A[i, j] / A[i, i]

    x = c.copy()
    for k in range(max_iter):
        x_new = c + np.dot(B, x)
        if np.linalg.norm(x_new - x) < eps:
            return x_new, k + 1
        x = x_new

    return x, max_iter


def seidel_method(A, b, eps=0.01, max_iter=100):
    n = len(A)
    x = np.zeros(n)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A[i, :i], x[:i])
            sum2 = np.dot(A[i, i + 1:], x_old[i + 1:])
            x[i] = (b[i] - sum1 - sum2) / A[i, i]

        if np.linalg.norm(x - x_old) < eps:
            return x, k + 1

    return x, max_iter