import numpy as np

# Рівняння 1: x1^2 - 2 = 0  =>  x1 = sqrt(2) або x1 = -sqrt(2)
def f1(x1):
    return x1**2 - 2

# Рівняння 2: cos(x1) - x2 = 0  =>  x2 = cos(x1)
def f2(x1):
    return np.cos(x1)


def phi_rosenbrock(X):
    """Цільова функція Розенброка"""
    return 100 * (X[0]**2 - X[1])**2 + (X[0] - 1)**2


def exploratory_search(phi, x_base, alpha):
   
    x = np.copy(x_base).astype(float)
    n = len(x)
    
    for i in range(n):
        f_current = phi(x)
              
        x[i] += alpha
        if phi(x) < f_current:
            continue  
                 
        x[i] -= 2 * alpha
        if phi(x) < f_current:
            continue 
           
    
        x[i] += alpha
        
    return x


def hooke_jeeves_to_file(phi, x0, filename, alpha=0.5, beta=0.5, eps=1e-6, max_iter=1000):
   
    x0 = np.array(x0, dtype=float)
    x_old = np.copy(x0)
    
   
    with open(filename, "w", encoding="utf-8") as f:
        f.write("Ітерація\tX1\t\tX2\t\tЗначення функції Phi(X)\n")
        f.write(f"0\t\t{x_old[0]:.6f}\t{x_old[1]:.6f}\t{phi(x_old):.6f}\n")
        
        x_new = exploratory_search(phi, x_old, alpha)
        
        iteration = 0
        step_count = 0  # Рахівник успішних кроків (змін базисної точки)
        
        while alpha > eps and iteration < max_iter:
            iteration += 1
            
            if not np.allclose(x_new, x_old, atol=1e-11):
                step_count += 1
                f.write(f"{step_count}\t\t{x_new[0]:.6f}\t{x_new[1]:.6f}\t{phi(x_new):.6f}\n")
                              
                direction = x_new - x_old
                x_pattern = x_new + direction
                
                x_pattern_explored = exploratory_search(phi, x_pattern, alpha)
                
                if phi(x_pattern_explored) < phi(x_new):
                    x_old = np.copy(x_new)
                    x_new = np.copy(x_pattern_explored)
                else:
                    x_old = np.copy(x_new)
                    x_new = exploratory_search(phi, x_old, alpha)
            else:
                alpha *= beta
                x_new = exploratory_search(phi, x_old, alpha)
                
    return x_old, phi(x_old), step_count
