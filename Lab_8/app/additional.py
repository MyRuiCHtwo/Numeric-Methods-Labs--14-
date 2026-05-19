import math
import os


def f(x: float) -> float:
    """Задана трансцендентна функція: F(x) = sin(x) - 0.5*x"""
    return math.sin(x) - 0.5 * x

def df(x: float) -> float:
  
    return math.cos(x) - 0.5

def d2f(x: float) -> float:
    
    return -math.sin(x)

def phi(x: float) -> float:
  
    tau = 1.0  
    return x + tau * f(x)


def tabulate_function(a: float, b: float, filename: str, h: float) -> list:
    
    results = []
    x = a
    
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(" x \t\t F(x)\n")
        while x <= b + 1e-9:
            fx = f(x)
            results.append((x, fx))
            fh.write(f"{x:.4f} \t {fx:.6f}\n")
            x += h
            
    return results

def find_initial_roots(tabulated_data: list) -> tuple:
  
    root_inc = None  # Корінь на ділянці зростання
    root_dec = None  # Корінь на ділянці спадання
    
    for i in range(len(tabulated_data) - 1):
        x1, y1 = tabulated_data[i]
        x2, y2 = tabulated_data[i+1]
              
        if y1 * y2 <= 0:
            approx_root = (x1 + x2) / 2.0
            derivative = df(approx_root)
                      
            if derivative > 0 and root_inc is None:
                root_inc = approx_root
            elif derivative < 0 and root_dec is None:
                root_dec = approx_root
                
        if root_inc is not None and root_dec is not None:
            break
            
    return root_inc, root_dec


def read_polynomial_coefficients(filename: str) -> list:
   
    coeffs = []
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Файл {filename} не знайдено.")
        
    with open(filename, 'r', encoding='utf-8') as fh:
        for line in fh:
            # вилуч порожні рядки
            parts = line.strip().split()
            for part in parts:
                try:
                    coeffs.append(float(part))
                except ValueError:
                    pass
    return coeffs

def horner_scheme(coeffs: list, x: float) -> tuple:
    
    m = len(coeffs) - 1
    
   
    b = [0.0] * (m + 1)
    b[m] = coeffs[m]
    for i in range(m - 1, -1, -1):
        b[i] = coeffs[i] + x * b[i + 1]
        
    f_val = b[0]
    
  
    if m >= 1:
        c = [0.0] * (m + 1)
        c[m] = b[m]
        for i in range(m - 1, 0, -1): 
            c[i] = b[i] + x * c[i + 1]
        df_val = c[1]
    else:
        df_val = 0.0
        
    return f_val, df_val
