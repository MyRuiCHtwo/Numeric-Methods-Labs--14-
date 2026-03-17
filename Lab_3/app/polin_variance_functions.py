import numpy as np


def polynomial(x, coef):
    y_poly = np.zeros(len(x))
    x = np.array(x)

    for i in range(len(coef)):
        y_poly += coef[i] * (x ** i)

    return y_poly


def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)
