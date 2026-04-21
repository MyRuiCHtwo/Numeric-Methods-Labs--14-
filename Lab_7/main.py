import numpy as np

from app.nm_realiz.functions import B_solve, matrix_norm_inf, meth_simple_iteration, math_Yacobi, meth_Gauss_Zeidelya

XI = 2.5
N = 100
X0 = np.full(N, 1.0)
# X0 = np.zeros(N)

def read_matrix_from_file(filenameA, filenameB):
    A = np.loadtxt(filenameA)
    B = np.loadtxt(filenameB)

    return A, B


def main():
    # A = np.random.uniform(1, 100, (N, N))
    # for i in range(N):
    #     A[i, i] += 5000.0
  
    
    # B_solve(A, XI, N)

    A, B = read_matrix_from_file("Lab_7/app/data/A.txt", "Lab_7/app/data/B.txt")
    
    tau = 1.0 / matrix_norm_inf(N, A)
    X_si, max_iter = meth_simple_iteration(N, A, B, X0, tau)


    X_ya, max_iter = math_Yacobi(N, A, B, X0)

    X_gz, max_iter = meth_Gauss_Zeidelya(N, A, B, X0)

    for i in range(N):
        print(f"X_si[{i}] = {X_gz[i]:.18f}")
    print(f"\n Simple Iteration Method converged in {max_iter} iterations.")

    


if __name__ == "__main__":
    main()