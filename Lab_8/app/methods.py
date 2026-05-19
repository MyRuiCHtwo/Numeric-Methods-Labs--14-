import numpy as np


def method_simple_iteration(f, x0: float, eps: float, max_iter: int = 1000) -> tuple:
    
    tau = -0.6 if f(x0 + 0.1) > f(x0) else 0.6 
    x_curr = x0
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        x_next = x_curr + tau * f(x_curr)
        
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, iterations
        x_curr = x_next
        
    return x_curr, iterations

def method_newton(f, df, x0: float, eps: float, max_iter: int = 100) -> tuple:
   
    x_curr = x0
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        f_val = f(x_curr)
        df_val = df(x_curr)
        
        if abs(df_val) < 1e-12:
            break
            
        x_next = x_curr - f_val / df_val
        
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, iterations
        x_curr = x_next
        
    return x_curr, iterations

def method_chebyshev(f, df, d2f, x0: float, eps: float, max_iter: int = 100) -> tuple:
   
    x_curr = x0
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        f_val = f(x_curr)
        df_val = df(x_curr)
        d2f_val = d2f(x_curr)
        
        if abs(df_val) < 1e-12:
            break
            
      
        correction = (f_val / df_val) + 0.5 * (f_val**2 * d2f_val) / (df_val**3)
        x_next = x_curr - correction
        
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, iterations
        x_curr = x_next
        
    return x_curr, iterations

def method_chord(f, x0: float, eps: float, max_iter: int = 100) -> tuple:
   
    x_prev = x0
    x_curr = x0 + 0.1
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        f_curr = f(x_curr)
        f_prev = f(x_prev)
        
        if abs(f_curr - f_prev) < 1e-12:
            break
            
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        if abs(f(x_next)) < eps and abs(x_next - x_curr) < eps:
            return x_next, iterations
            
        x_prev = x_curr
        x_curr = x_next
        
    return x_curr, iterations

def method_parabola(f, x0: float, eps: float, max_iter: int = 100) -> tuple:
   
    x = np.array([x0 - 0.1, x0 + 0.1, x0], dtype=complex)
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        f0, f1, f2 = f(x[0]), f(x[1]), f(x[2])
        
     
        h1 = x[1] - x[0]
        h2 = x[2] - x[1]
        
        if abs(h1) < 1e-12 or abs(h2) < 1e-12:
            break
            
        d1 = (f1 - f0) / h1
        d2 = (f2 - f1) / h2
        d = (d2 - d1) / (x[2] - x[0])
        
     
        w = d2 + h2 * d
        
      
        sqrt_discr = np.sqrt(w**2 - 4 * f2 * d)
        if abs(w - sqrt_discr) > abs(w + sqrt_discr):
            denom = w - sqrt_discr
        else:
            denom = w + sqrt_discr
            
        if abs(denom) < 1e-12:
            break
            
        dx = -2 * f2 / denom
        x_next = x[2] + dx
        
        if abs(f(x_next)) < eps and abs(dx) < eps:
          
            res_root = x_next.real if abs(x_next.imag) < 1e-8 else x_next
            return res_root, iterations
            
       
        x[0], x[1], x[2] = x[1], x[2], x_next
        
    res_root = x[2].real if abs(x[2].imag) < 1e-8 else x[2]
    return res_root, iterations

def method_inverse_interpolation(f, x0: float, eps: float, max_iter: int = 100) -> tuple:
   
    x = np.array([x0 - 0.1, x0, x0 + 0.1])
    iterations = 0
    
    for _ in range(max_iter):
        iterations += 1
        y = np.array([f(x[0]), f(x[1]), f(x[2])])
        
    
        if len(np.unique(y)) < 3:
            break
            
      
        x_next = (
            (y[1] * y[2]) / ((y[0] - y[1]) * (y[0] - y[2])) * x[0] +
            (y[0] * y[2]) / ((y[1] - y[0]) * (y[1] - y[2])) * x[1] +
            (y[0] * y[1]) / ((y[2] - y[0]) * (y[2] - y[1])) * x[2]
        )
        
        if abs(f(x_next)) < eps and abs(x_next - x[1]) < eps:
            return x_next, iterations
            
       
        distances = np.abs(x - x_next)
        max_idx = np.argmax(distances)
        x[max_idx] = x_next
        
    return x[1], iterations


def method_newton_horner(coeffs: list, x0: float, eps: float, max_iter: int = 100) -> tuple:
  
    from app.additional import horner_scheme
    
    x_curr = x0
    iterations = 0
    c_arr = np.array(coeffs, dtype=float)
    
    for _ in range(max_iter):
        iterations += 1
       
        f_val, df_val = horner_scheme(c_arr.tolist(), x_curr)
        
        if abs(df_val) < 1e-12:
            break
            
        x_next = x_curr - f_val / df_val
        
        if abs(horner_scheme(c_arr.tolist(), x_next)[0]) < eps and abs(x_next - x_curr) < eps:
            return x_next, iterations
        x_curr = x_next
        
    return x_curr, iterations

def method_lin(coeffs: list, eps: float, max_iter: int = 500) -> tuple:
  
    a = np.array(coeffs, dtype=float)
    m = len(a) - 1
    
    if m < 3:
        raise ValueError("Метод Ліна реалізовано для многочленів степеня 3 і вище.")
        
   
    s = a[m-1] / a[m]
    t = a[m-2] / a[m]
    
    iterations = 0
    b = np.zeros(m + 1)
    
    for _ in range(max_iter):
        iterations += 1
        
       
        b[m] = a[m]
        b[m-1] = a[m-1] - s * b[m]
        
        for k in range(m - 2, -1, -1):
            b[k] = a[k] - s * b[k+1] - t * b[k+2]
            
     
        denom = b[2]**2 - b[1]*b[3] if m > 3 else b[2] 
        
        if abs(b[2]) < 1e-12:
            break
            
       
        s_next = (a[1] - b[1]) / b[2]
        t_next = a[0] / b[2]
        
      
        if abs(s_next - s) < eps and abs(t_next - t) < eps:
            s, t = s_next, t_next
            break
            
        s, t = s_next, t_next
            
    discr = np.iscomplex(s**2 - 4*t)
    roots = np.roots([1.0, s, t])
    
    return roots, iterations
