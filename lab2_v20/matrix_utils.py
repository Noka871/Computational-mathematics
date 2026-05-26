import numpy as np

def print_system(A, b):
    n = len(A)
    print("\nСистема уравнений:")
    for i in range(n):
        row = ""
        for j in range(n):
            if A[i, j] != 0:
                if j == 0:
                    row += f"{A[i, j]:.3f}x{j+1}"
                else:
                    sign = " + " if A[i, j] > 0 else " - "
                    row += f"{sign}{abs(A[i, j]):.3f}x{j+1}"
        row += f" = {b[i]:.3f}"
        print(row)

def check_convergence(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag <= row_sum:
            return False
    return True

def residual(A, x, b):
    return np.linalg.norm(np.dot(A, x) - b)