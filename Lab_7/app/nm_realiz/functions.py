import numpy as np


def B_solve(A, xi, n) -> None:
    B = np.zeros(n) 

    for i in range(n):
        B[i] = sum(A[i, j] * xi for j in range(n))

    np.savetxt("Lab_7/app/data/A.txt", A, fmt="%.18f")
    np.savetxt("Lab_7/app/data/B.txt", B, fmt="%.18f")


def A_B_product(n, A, B): 
    AB = np.zeros(n)
    for i in range(n):
        AB[i] = sum(A[i, j] * B[j] for j in range(n))

    return AB


def solve_vector_norm(n, X, Y):
    max_val = 0.0
    for i in range(n):
        diff = abs(Y[i] - X[i])
        if diff > max_val:
            max_val = diff
    
    return max_val
 

def matrix_norm_inf(n, A):
    max_row_sum = 0.0
    for i in range(n):
        current_row_sum = sum(abs(A[i, j]) for j in range(n))
        if current_row_sum > max_row_sum:
            max_row_sum = current_row_sum

    return max_row_sum


def meth_simple_iteration(n, A, B, X0, tau, eps=1e-14, max_iter=100000):
    X1 = np.copy(X0)
    for k in range(1, max_iter + 1):
        x_new = np.zeros(n)
        for i in range(n):
           sum_ax = sum(A[i, j] * X1[j] for j in range(n))
           x_new[i] = X1[i] - tau * sum_ax + tau * B[i]

        if solve_vector_norm(n, x_new, X1) < eps:
            return x_new, k + 1
        X1 = x_new

    return X1, max_iter


def math_Yacobi(n, A, B, X0, eps=1e-14, max_iter=100000):
    X1 = np.copy(X0)
    for k in range(1, max_iter+  1):
        x_new = np.zeros(n)
        for i in range(n):
            sum_ax = sum(A[i, j] * X1[j] for j in range(n) if j != i)
            x_new[i] = (B[i] - sum_ax) / A[i, i]

        if solve_vector_norm(n, x_new, X1) < eps:
            return x_new, k + 1
        X1 = x_new

    return X1, max_iter


def meth_Gauss_Zeidelya(n, A, B, X0, eps=1e-14, max_iter=100000):
    X1 = np.copy(X0)
    for k in range(1, max_iter + 1):
        x_new = np.copy(X1)
        for i in range(n):
            sum_ax = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (B[i] - sum_ax) / A[i, i]

        if solve_vector_norm(n, x_new, X1) < eps:
            return x_new, k + 1
        X1 = x_new

    return X1, max_iter
