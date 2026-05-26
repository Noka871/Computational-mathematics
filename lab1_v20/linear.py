import numpy as np
import time

A = np.array([
    [2.74, -1.18, 3.17],
    [1.12, 0.83, -2.16],
    [0.81, 1.27, 0.76]
], dtype=float)

b = np.array([2.18, 1.15, 3.23], dtype=float)

def gauss(A, b):
    start = time.time()
    n = len(b)
    Ab = np.hstack([A, b.reshape(-1, 1)])
    
    for i in range(n):
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        Ab[[i, max_row]] = Ab[[max_row, i]]
        
        Ab[i] = Ab[i] / Ab[i, i]
        
        for k in range(i + 1, n):
            Ab[k] -= Ab[k, i] * Ab[i]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])
    
    residual = np.linalg.norm(np.dot(A, x) - b)
    return {
        "solution": x,
        "residual": residual,
        "time": time.time() - start
    }

def orthogonalization(A, b):
    start = time.time()
    n = len(b)
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    
    y = np.dot(Q.T, b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]
    
    residual = np.linalg.norm(np.dot(A, x) - b)
    return {
        "solution": x,
        "residual": residual,
        "time": time.time() - start
    }

def simple_iteration_linear(A, b, eps=1e-6, max_iter=1000):
    """Simple iteration with proper convergence conditions"""
    start = time.time()
    n = len(b)
    x = np.zeros(n)
    
    # Transform system to ensure convergence: x = Bx + c
    # Use Jacobi iteration with relaxation
    D = np.diag(A)
    R = A - np.diag(D)
    
    # Check convergence condition
    B = -np.diag(1.0 / D) @ R
    spectral_radius = np.max(np.abs(np.linalg.eigvals(B)))
    
    if spectral_radius >= 1:
        # Scale the system to improve convergence
        scaling = 1.0 / np.max(np.abs(A), axis=1)
        A_scaled = A * scaling[:, np.newaxis]
        b_scaled = b * scaling
        D_scaled = np.diag(A_scaled)
        R_scaled = A_scaled - np.diag(D_scaled)
        omega = 0.5  # Relaxation parameter
    else:
        A_scaled = A
        b_scaled = b
        D_scaled = D
        R_scaled = R
        omega = 1.0
    
    for k in range(max_iter):
        # Jacobi iteration with relaxation: x_new = omega * D^{-1}(b - Rx) + (1-omega)*x
        x_new = omega * (b_scaled - np.dot(R_scaled, x)) / D_scaled + (1 - omega) * x
        
        if np.linalg.norm(x_new - x) < eps:
            x = x_new
            break
        
        x = x_new
    
    residual = np.linalg.norm(np.dot(A, x) - b)
    return {
        "solution": x,
        "iterations": k + 1,
        "residual": residual,
        "time": time.time() - start
    }

def seidel(A, b, eps=1e-6, max_iter=1000):
    """Improved Gauss-Seidel method"""
    start = time.time()
    n = len(b)
    x = np.zeros(n)
    
    # Make a copy to avoid modifying the original
    A_work = A.copy()
    b_work = b.copy()
    
    # Ensure diagonal elements are non-zero
    for i in range(n):
        if abs(A_work[i, i]) < 1e-10:
            # Swap with a row that has non-zero diagonal
            for j in range(i + 1, n):
                if abs(A_work[j, i]) > 1e-10:
                    A_work[[i, j]] = A_work[[j, i]]
                    b_work[[i, j]] = b_work[[j, i]]
                    break
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sum1 = np.dot(A_work[i, :i], x[:i])
            sum2 = np.dot(A_work[i, i+1:], x_old[i+1:])
            if abs(A_work[i, i]) > 1e-12:
                x[i] = (b_work[i] - sum1 - sum2) / A_work[i, i]
        
        if np.linalg.norm(x - x_old) < eps:
            break
    
    residual = np.linalg.norm(np.dot(A, x) - b)
    return {
        "solution": x,
        "iterations": k + 1,
        "residual": residual,
        "time": time.time() - start
    }
