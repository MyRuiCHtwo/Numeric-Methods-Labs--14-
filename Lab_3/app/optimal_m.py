import numpy as np

from app.polin_variance_functions import polynomial, variance
from app.tsqm_functions import form_matrix_A, form_vector_b, gauss_solve

def get_optimal_m(max_degree, x, y):
    variances = []
    for m in range(1, max_degree + 1):
        A = form_matrix_A(x, m)
        b = form_vector_b(x, y, m)
        coeffs = np.linalg.solve(A, b)
        y_approx = polynomial(x, coeffs)
        var = variance(y, y_approx)
        variances.append(var)

    optimal_m = variances.index(min(variances)) + 1


    return optimal_m, variances
