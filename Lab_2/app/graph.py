import matplotlib.pyplot as plt
import numpy as np

from app.newton_interpolation import get_newton_polynomial
from app.separated_differences import get_separated_differences


def plot_graph_fps(x, y):
    x_new = np.linspace(min(x), max(x), 1000)
   
    plt.figure(figsize=(12, 8))

    configs = [
        (5, 'blue', '-', 2),   # n=5: основна модель
        (10, 'green', '--', 1.5), # n=10: середня деталізація
        (20, 'red', ':', 1.5)    # n=20: високий степінь (ефект Рунге)
    ]

    for n, color, linestyle, linewidth in configs:
        x_n = x[:n]
        y_n = y[:n]

        coefs = get_separated_differences(x_n, y_n)
        y_new = [get_newton_polynomial(xi, x_n, coefs) for xi in x_new]

        plt.plot(x_new, y_new, label=f'Многочлен Ньютона (n={n})',color=color, linestyle=linestyle, linewidth=linewidth)
    plt.scatter(x, y, color='black', s=25, zorder=5, label='Експериментальні дані')
       

    plt.axhline(60, color='darkgreen', linestyle='-.', alpha=0.6, label='Поріг комфорту (60 FPS)')

    plt.ylim(-50, 250) # Обмежуємо по Y, щоб "дикі" коливання не стискали графік
    plt.title('Порівняльний аналіз інтерполяції FPS при різній кількості вузлів', fontsize=14)
    plt.xlabel('Кількість об\'єктів (n)', fontsize=12)
    plt.ylabel('FPS', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    plt.show()


def plot_errors(x, y):
    plt.figure(figsize=(10, 6))

    nodes = [5, 10, 20]
    colors = ['blue', 'green', 'red']

    for n, color in zip(nodes, colors):
        x_n = x[:n]
        y_n = y[:n]

        coefs = get_separated_differences(x_n, y_n)

        errors = []
        for i in range(len(x)):
            predicted = get_newton_polynomial(x[i], x_n, coefs)
            error = abs(predicted - y[i])
            errors.append(error)

        plt.plot(x, errors, label=f'Похибка моделі n={n}', color=color, marker='o', linestyle='-', alpha=0.8)
    
    plt.title('Порівняння абсолютних похибок для 5, 10 та 20 вузлів', fontsize=14)
    plt.xlabel('Кількість об\'єктів (n)', fontsize=12)
    plt.ylabel('Абсолютна похибка (FPS)', fontsize=12)
    
    # Використовуємо логарифмічну шкалу, бо похибка n=20 може бути гігантською
    plt.yscale('log') 
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.show()




 