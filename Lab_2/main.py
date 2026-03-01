import csv

from app.separated_differences import get_separated_differences
from app.newton_interpolation import get_newton_polynomial
from app.graph import plot_graph_fps, plot_errors

def read_csv(file_path):
    x, y = [], []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['fps']))

    return x, y


def main():
    file_path = 'Lab_2/data/fps_per_object20.csv'

    x, y = read_csv(file_path)

    # print("x:", x)
    # print("y:", y)

    # coef_for_newton = get_separated_differences(x, y)
    # print("Coefficients for Newton's Interpolation:", coef_for_newton)

    # n_target = 2000
    # prediction = get_newton_polynomial(n_target, x, coef_for_newton)
    # print(f"Predicted FPS for {n_target} objects: {prediction:.2f}")

    plot_graph_fps(x, y)
    plot_errors(x, y)



if __name__ == "__main__":
    main()












