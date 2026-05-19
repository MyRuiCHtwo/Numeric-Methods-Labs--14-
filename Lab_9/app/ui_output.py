import matplotlib.pyplot as plt
import numpy as np


def plot_sys(x_range, x_l1, x_l2, phi):
    plt.figure(figsize=(8, 5))
  
    plt.plot(x_range, x_l1, label="f1: x2 = x1² - 2", color="crimson", linewidth=2)
    plt.plot(x_range, x_l2, label="f2: x2 = cos(x1)", color="royalblue", linewidth=2)
  
    X0 = np.array([1.5, 0.2])
    plt.scatter(X0[0], X0[1], color="darkgreen", s=100, zorder=5, 
                label=f"Базисна точка X0 {tuple(X0)}")

    # Оформлення осей та сітки
    plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
    plt.axvline(0, color='black', linewidth=0.8, linestyle='-')
    plt.title("Cистема рівнянь для вибору X0")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.ylim(-3, 3)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper center")

    plt.show()
