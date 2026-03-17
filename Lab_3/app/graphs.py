import matplotlib.pyplot as plt
import numpy as np



def plot_aproximation(x, y, y_approx, m):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="black", label='Фактичні дані')
    plt.plot(x, y, color="red", label='Фактичні дані (лінія)')
    plt.plot(x, y_approx, color="blue", label=f'Апроксимація (m={m})')
    plt.title('Апроксимація температурних даних')
    plt.xlabel('Місяць')
    plt.ylabel('Температура')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

def plot_residals(x,y, y_approx):
    residuals = y - y_approx
    plt.figure(figsize=(10, 6))
    plt.bar(x, residuals, color="purple", label='Залишки (Похибки)', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f'Розподіл похибок (Залишки)')
    plt.xlabel('Місяць')
    plt.ylabel('Похибка (y_real - y_poly)')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)


def plot_dispersion(variances):
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, len(variances) + 1), variances, marker='o', color='green')
    plt.title('Залежність дисперсії від ступеня полінома')
    plt.xlabel('Ступінь полінома (m)')
    plt.ylabel('Дисперсія (Variance)')
    plt.xticks(range(1, len(variances) + 1))

    optimal_m = variances.index(min(variances)) + 1
    plt.annotate(f'Оптимальне m={optimal_m}', 
             xy=(optimal_m, variances[optimal_m-1]), 
             xytext=(optimal_m + 0.5, variances[optimal_m-1] + 10),
             arrowprops=dict(facecolor='black', shrink=0.05))
    plt.grid(True)
    plt.show(block=True)


def plot_prognose(x_future, y_future, x, y):
   

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="black")
    plt.plot(x, y, color="red", label='Фактичні дані')
    plt.scatter(x_future, y_future, color="blue", marker='o')
    plt.plot(x_future, y_future, color="blue", linestyle='--', label='Прогноз')

   
    plt.title('Прогноз температурних даних')
    plt.xlabel('Місяць')
    plt.ylabel('Температура')
    plt.legend()
    plt.grid(True)
    plt.show(block=True)
    