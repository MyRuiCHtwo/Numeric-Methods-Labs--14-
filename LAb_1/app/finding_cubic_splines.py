import numpy as np



def f(x):
    return 1500 + 300*np.sin(x/300)


def tabluate_function(distances, elevations):
    full_x = np.array(distances)
    full_y = np.array(elevations)

    # Вказуємо бажану кількість вузлів (наприклад, 10)
    real_distances = len(full_x)

    target_nodes = 10

    n_nodes = min(target_nodes, real_distances)

    # Вибираємо індекси так, щоб вони були рівномірно розподілені
    indices = np.linspace(0, real_distances - 1, n_nodes, dtype=int)

    # Створюємо нові вузли для розрахунку сплайна
    x = full_x[indices]
    y = full_y[indices]
    N = len(x) - 1

    h = np.zeros(N + 1)
    for i in range(1, N + 1):
        h[i] = x[i] - x[i-1] 

    with open ("Lab_1/data/input.txt", 'w', encoding='utf-8') as fh:
        for i in range(N + 1):
            fh.write(f"{i:2d} | {x[i]:.6f} | {y[i]:.6f}\n")
    return N


def progonka(y, h, N, c):
    alpha, beta, hamma, delta, A, B = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
    alpha[1], hamma[1], delta[1] = 0.0, 0.0, 0.0
    beta[1] = 1.0

    for i in range(2, N):
        alpha[i] = h[i - 1]
        beta[i] = 2*(h[i-1]+ h[i])
        hamma[i] = h[i]
        delta[i] = 3 * ((y[i] - y[i-1]) / h[i] - (y[i-1] - y[i-2]) / h[i-1])

    hamma[N] = 0.0

    A[1] = -hamma[1] / beta[1]
    B[1] = delta[1] / beta[1]
    for i in range(2, N):
        A[i] = -hamma[i] / (alpha[i] * A[i-1] + beta[i])
        B[i] = (delta[i] - alpha[i] * B[i-1]) / (alpha[i] * A[i-1] + beta[i])

   
    for i in range(N - 1, 0, -1):
        c[i] = A[i] * c[i+1] + B[i]

    c[0] = 0.0
    c[N] = 0.0
    
    return c

    
def main(N):
    x, y, h = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)
    a, b, c, d = np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1), np.zeros(N + 1)

    xm, ym = np.zeros(20*N + 1), np.zeros(20*N + 1)

    with open ("Lab_1/data/input.txt", 'r', encoding='utf-8') as fh:
        for i in range(N + 1):
            line = fh.readline().split('|')
            x[i] = float(line[1])
            y[i] = float(line[2])
            if i > 0:
                h[i] = x[i] - x[i-1]

    # Populate xm and ym with interpolation points and their function values
    for i in range(20 * N + 1):
        xm[i] = x[0] + i * (x[N] - x[0]) / (20 * N)
        ym[i] = f(xm[i])

    c = progonka(y, h, N, np.zeros(N + 1))
    # for i in range(c.size):
    #     print(f"c[{i}] = {c[i]:.6f}")

    for i in range(0, N):
        a[i] = y[i]
        b[i] = (y[i+1] - y[i]) / h[i+1] - h[i+1] * (2*c[i] + c[i+1]) / 3
        d[i] = (c[i+1] - c[i]) / (3*h[i+1])

  

    # for i in range(1, N + 1):
    #     print(f"a[i] = {a[i]:.6f}, b[i] = {b[i]:.6f}, d[i] = {d[i]:.6f}")

    with open("Lab_1/data/output.txt", 'w', encoding='utf-8') as fh:
        for i in range(20*N + 1):
           # Find the correct interval for xm[i]
            j = np.searchsorted(x, xm[i]) - 1
            j = max(0, min(j, N-1))

            dx = xm[i] - x[j]
            s = a[j] + b[j]*dx + c[j]*(dx**2) + d[j]*(dx**3)
            eps = abs(s-ym[i])

            fh.write(f"{i:3d} | {xm[i]:.6f} | {ym[i]:.6f} | {s:.6f} | {eps:.6f}\n")

