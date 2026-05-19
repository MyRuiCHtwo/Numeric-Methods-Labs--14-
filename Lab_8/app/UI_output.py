import numpy as np
import matplotlib.pyplot as plt


def print_transcendental_header(root_type: str):
 
    print(f"\n" + "="*75)
    print(f" Результати уточнення кореня на ділянці: {root_type.upper()}")
    print("="*75)
    print(f"{'Метод розрахунку':<30} | {'Знайдений корінь (x)':<22} | {'Ітерацій':<8}")
    print("-"*75)

def print_transcendental_row(method_name: str, root: float, iterations: int):
   
    print(f"{method_name:<30} | {root:<22.10f} | {iterations:<8}")

def print_polynomial_results(n_root: float, n_iter: int, lin_roots: np.ndarray, lin_iter: int):
  
    print(f"\n" + "="*75)
    print(" Результати розв'язання алгебраїчного рівняння третього порядку")
    print("="*75)
    
    print(f"1. Метод Ньютона (зі схемою Горнера) для дійсного кореня:")
    print(f"   - Дійсний корінь x1 = {n_root:.10f}")
    print(f"   - Кількість ітерацій: {n_iter}")
    print("-"*75)
    
    print(f"2. Метод Ліна для знаходження комплексно-спряжених коренів:")
    print(f"   - Кількість ітерацій: {lin_iter}")
    for i, root in enumerate(lin_roots, start=2):
        if np.iscomplex(root):
            print(f"   - Комплексний корінь x{i} = {root.real:.6f} + {root.imag:.6f}i" if root.imag >= 0 
                  else f"   - Комплексний корінь x{i} = {root.real:.6f} - {abs(root.imag):.6f}i")
        else:
            print(f"   - Дійсний корінь x{i} = {root.real:.6f}")
    print("="*75 + "\n")


def plot_transcendental_function(f, a: float, b: float, roots: list, root_labels: list):
  
    x_val = np.linspace(a - 0.5, b + 0.5, 500)
    y_val = [f(x) for x in x_val]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_val, y_val, label=r'$F(x) = \sin(x) - 0.5x$', color='royalblue', lw=2)
    
   
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
       
    colors = ['crimson', 'darkgreen']
    for root, label, color in zip(roots, root_labels, colors):
        if root is not None:
            plt.scatter(root, f(root), color=color, s=80, zorder=5,
                        label=f"Корінь ({label}): x ≈ {root:.4f}")
            
    plt.title('Локалізація та уточнення коренів трансцендентної функції', fontsize=14, pad=15)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('F(x)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    plt.show()

def plot_polynomial_function(coeffs: list, x_start: float, x_end: float, real_root: float, complex_roots: np.ndarray):
    
    poly_coeffs_reversed = coeffs[::-1]
    
    x_val = np.linspace(x_start, x_end, 500)
    y_val = np.polyval(poly_coeffs_reversed, x_val)
    
    plt.figure(figsize=(10, 6))
       
    poly_label = f"$P_3(x) = {coeffs[3]}x^3 + {coeffs[2]}x^2 + {coeffs[1]}x + {coeffs[0]}$"
    plt.plot(x_val, y_val, label=poly_label, color='darkorange', lw=2)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
     
    if real_root is not None:
        plt.scatter(real_root, 0, color='purple', s=80, zorder=5, 
                    label=f'Дійсний корінь: x = {real_root:.4f}')
          
    c_text = "Комплексні корені рівняння:\n"
    for i, r in enumerate(complex_roots):
        if np.iscomplex(r):
            c_text += f"x{i+2} = {r.real:.3f} ± {abs(r.imag):.3f}i\n"
            break
            
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.gca().text(0.05, 0.15, c_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    plt.title('Графік алгебраїчного рівняння третього порядку', fontsize=14, pad=15)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('P(x)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=11, loc='upper center')
    plt.show()
    