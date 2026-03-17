import numpy as np

def normalize_x(x):
    """Normalize x values to [-1, 1] range for better numerical stability."""
    x_array = np.array(x)
    x_min = np.min(x_array)
    x_max = np.max(x_array)
    x_range = x_max - x_min
    if x_range == 0:
        return x_array, x_min, 1.0
    x_normalized = 2 * (x_array - x_min) / x_range - 1
    return x_normalized, x_min, x_range

def denormalize_x(x_normalized, x_min, x_range):
    """Convert normalized x values back to original range."""
    return (x_normalized + 1) * x_range / 2 + x_min

def form_matrix_A(x, m):
    A = np.zeros((m+1, m+1))
    for i in range(m+1):
        for j in range(m+1):
            A[i][j] = sum(x_k**(i+j) for x_k in x)
    return A


def form_vector_b(x, y, m):
    b = np.zeros(m+1)

    for i in range(m+1):
        b[i] = sum(y_k * x_k**i for x_k, y_k in zip(x, y))
    
    return b


def gauss_solve(A, b):
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)
    
    # Forward elimination with partial pivoting
    for k in range(n):
        # Find pivot
        max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[max_row][k]) < 1e-14:
            raise ValueError(f"Matrix is singular or nearly singular at column {k}")
        
        # Swap rows
        A[k], A[max_row] = A[max_row].copy(), A[k].copy()
        b[k], b[max_row] = b[max_row], b[k]
        
        # Eliminate below
        for i in range(k+1, n):
            factor = A[i][k] / A[k][k]
            A[i] = A[i] - factor * A[k]
            b[i] = b[i] - factor * b[k]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        if abs(A[i][i]) < 1e-14:
            raise ValueError(f"Matrix is singular or nearly singular at row {i}")
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i+1, n))) / A[i][i]

    return x


