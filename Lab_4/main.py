import numpy as np

def M(t):
       
        return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

def dM_exact(t):
  
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

def central_diff(f, x, h):
       
        return (f(x + h) - f(x - h)) / (2 * h)

def main():
   
    print("              Завдання 1: Аналітичне розв'язання")
    print("-" * 70)
   
    t0 = 1.0
    exact_val = dM_exact(t0)
    print(f"Точне значення похідної в точці t0={t0}: {exact_val}")
    print("-" * 70)
   
    h_values = np.logspace(-20, 3, 1000)
    errors = []

    for h_val in h_values:
        approx_val = central_diff(M, t0, h_val)
        errors.append(abs(approx_val - exact_val))

    min_error_idx = np.argmin(errors)
    h0 = h_values[min_error_idx]
    R0 = errors[min_error_idx]

    print("\n      Завдання 2: Дослідження залежності похибки від кроку h:")
    print("-" * 70)

    print(f"Оптимальний крок h0: {h0:.1e}")
    print(f"Досягнута точність R0: {R0:.1e}")
    print("-" * 70)
  
    h = 1e-3
   
    y_prime_h = central_diff(M, t0, h)
    y_prime_2h = central_diff(M, t0, 2 * h)

  
    R1 = abs(y_prime_h - exact_val)

    print("\n      Завдання 3-5: Обчислення похідної та похибки для h=1e-3")
    print("-" * 70)

    print(f"Похибка при заданому кроці h={h} (R1): {R1:.1e}")
    print("-" * 70)

    y_prime_R = y_prime_h + (y_prime_h - y_prime_2h) / 3
    R2 = abs(y_prime_R - exact_val)

    print("\n         Завдання 6: Метод Рунге-Ромберга")
    print("-" * 70)

    print(f"Уточнене значення (Рунге-Ромберг): {y_prime_R}")
    print(f"Похибка (R2): {R2:.1e}")
    print("-" * 70)

    y_prime_4h = central_diff(M, t0, 4 * h)

    denominator = 2 * y_prime_2h - (y_prime_4h + y_prime_h)

    print("\n         Завдання 7: Метод Ейткена")
    print("-" * 70)

    if denominator != 0:
        y_prime_E = ((y_prime_2h**2) - y_prime_4h * y_prime_h) / denominator
        R3 = abs(y_prime_E - exact_val)
        
        ratio = abs((y_prime_4h - y_prime_2h) / (y_prime_2h - y_prime_h))
        p = (1 / np.log(2)) * np.log(ratio)
        
        print(f"Уточнене значення похідної: {y_prime_E}")
        print(f"Порядок точності формули (p): {p:.4f}")
        print(f"Похибка апроксимації (R3): {R3:.1e}")
    else:
        print("7. Метод Ейткена: ділення на нуль.")
    
    print("-" * 70)
        

if __name__ == "__main__":
    main()
    