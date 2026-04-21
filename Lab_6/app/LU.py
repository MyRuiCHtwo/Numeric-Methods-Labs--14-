import numpy as np





def B_solve(A, xi, n) -> None:
    B = np.zeros(n) 

    for i in range(n):
        B[i] = sum(A[i, j] * xi for j in range(n))

    np.savetxt("Lab_6/app/data/A.txt", A, fmt="%.18f")
    np.savetxt("Lab_6/app/data/B.txt", B, fmt="%.18f")
       
    


def L_U_solve(n, A) -> None:

    L = np.zeros_like(A)
    U = np.zeros_like(A)

    for i in range(n):
        U[i, i] = 1.0

    for k in range(n):
        for j in range(k, n):
            L[j, k] = A[j, k] - sum(L[j, m] * U[m, k] for m in range(k))
        
        for i in range(k+1, n):
            sum_u = sum(L[k, m] * U[m, i] for m in range(k))
            if L[k, k] == 0:
                raise ValueError("Zero pivot encountered at index {}".format(k))
            
            U[k, i] = (A[k, i] - sum_u) / L[k, k]

    np.savetxt("Lab_6/app/data/L.txt", L, fmt="%.18f")
    np.savetxt("Lab_6/app/data/U.txt", U, fmt="%.18f")

   


def solve_SLAE(n, B, L, U):
    Z = np.zeros(n)
    res = np.zeros(n)

    for k in range(n):
        sum_lu = sum(L[k, j] * Z[j] for j in range(k))
        Z[k] = (B[k] - sum_lu) / L[k, k]

    for k in range(n-1, -1, -1):
        sum_lu = sum(U[k, j] * res[j] for j in range(k+1, n))
        res[k] = Z[k] - sum_lu

    np.savetxt("Lab_6/app/data/X_calc.txt", res, fmt="%.18f")
    return res



def A_X_product(n, A, X): 
    AX = np.zeros(n)
    for i in range(n):
        AX[i] = sum(A[i, j] * X[j] for j in range(n))

    return AX


def solve_vector_norm(n, X, x_ideal=2.5):
    max_val = 0.0
    for i in range(n):
        diff = abs(X[i] - x_ideal)
        if diff > max_val:
            max_val = diff
    
    return max_val


def LU_error(n, A, X_calc, B):
    AX = A_X_product(n, A, X_calc)
    error = 0.0
    for i in range(n):
        diff = abs(B[i] - AX[i])
        if diff > error:
            error = diff

    return error

















        

