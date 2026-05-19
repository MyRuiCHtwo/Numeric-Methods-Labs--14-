import numpy as np

import app.methods
import app.ui_output


def main():
    file_output = "Lab_9/data/trajectory_hooke.txt"
    phi = app.methods.phi_rosenbrock
    x_r = np.linspace(-3, 3, 500)
 
    x_l1 = app.methods.f1(x_r)
    x_l2 = app.methods.f2(x_r)
  
    app.ui_output.plot_sys(x_r, x_l1, x_l2, phi)

    X0 = np.array([-1.2, 0.0])

    best_x, best_phi, total_steps = app.methods.hooke_jeeves_to_file(
        phi, X0, filename=file_output)
    
    print("\n--- Результати оптимізації ---")
    print("  Найкраще знайдене рішення:", best_x)
    print("  Значення цільової функції в найкращому рішенні:", round(best_phi, 6))
    print(f"\n  Кількість кроків на траєкторії спуску: {total_steps}")
    print(f"\n  Траєкторію успішно збережено у файл!")





if __name__ == "__main__":
    main()