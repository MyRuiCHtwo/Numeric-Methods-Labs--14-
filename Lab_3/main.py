import csv

import numpy as np

from app.optimal_m import get_optimal_m
from app.polin_variance_functions import polynomial, variance
from app.tsqm_functions import form_matrix_A, form_vector_b, gauss_solve, normalize_x, denormalize_x
from app.graphs import plot_aproximation, plot_residals, plot_dispersion, plot_prognose


def read_csv(file_path):
    x, y = [], []
    with open(file_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))

    return x, y

         
def main():
    x, y = read_csv("Lab_3/data/temp_per_month.csv")
    target_m = 5
    m, variances = get_optimal_m(target_m, x, y)
    # print(f"Optimal m: {m}")
    # print(f"Variances: {variances}")

    # Normalize x for better numerical stability
    x_norm, x_min, x_range = normalize_x(x)
    
    A = form_matrix_A(x_norm, m)
    b = form_vector_b(x_norm, y, m)
    coef = gauss_solve(A, b)
    y_approx = polynomial(x_norm, coef)


    x_future = [25]
    x_future_norm = (np.array(x_future) - x_min) * 2 / x_range - 1
    y_future = polynomial(x_future_norm, coef)
   

    print(f"Predicted temperatures for months {x_future}: {y_future}")

    # plot_aproximation(x, y, y_approx, m)
    # plot_residals(x, y, y_approx)
    # plot_dispersion(variances)
    plot_prognose(x_future, y_future, x, y)


if __name__ == "__main__":
    main()

  