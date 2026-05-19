import numpy as np


X0: float = 0.0
Y0: float = 1.0
X_END: float = 1.0

H_DEFAULT: float = 0.1

EPS_DEFAULT: float = 1e-4

RK4_ORDER: int = 4

C_RUNGE: float = 0.5


def ode_f(x: float, y: float) -> float:
    return x + y


def exact(x: float | np.ndarray) -> float | np.ndarray:
    return 2.0 * np.exp(x) - x - 1.0


def adams2(
    x0: float = X0,
    y0: float = Y0,
    x_end: float = X_END,
    h: float = H_DEFAULT,
            ) -> tuple[np.ndarray, np.ndarray]:
    
    x1 = x0 + h
    y1 = _rk4_step(x0, y0, h)

    xs = [x0, x1]
    ys = [y0, y1]

    f_prev = ode_f(x0, y0)
    f_curr = ode_f(x1, y1)
    x, y   = x1, y1

    while x + h <= x_end + 1e-12:
       
        y_pred = y + h / 2.0 * (3.0 * f_curr - f_prev)
        x_next = round(x + h, 14)
        f_pred = ode_f(x_next, y_pred)
      
        y_corr = y + h / 2.0 * (f_pred + f_curr)
       
        f_prev, f_curr = f_curr, ode_f(x_next, y_corr)
        x, y = x_next, y_corr
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


def auto_step_adams2(
    x0: float = X0,
    y0: float = Y0,
    x_end: float = X_END,
    h0: float = H_DEFAULT,
    eps: float = EPS_DEFAULT,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    ADAMS2_ERROR_COEFF = 3.0        

    x, y, h = x0, y0, h0

  
    x1 = x0 + h
    y1 = _rk4_step(x0, y0, h)

    xs = [x0, x1]
    ys = [y0, y1]
    hs = [h, h]

    f_prev = ode_f(x0, y0)
    f_curr = ode_f(x1, y1)
    x, y   = x1, y1

    while x < x_end - 1e-12:
        h = min(h, x_end - x)

        y_pred = y + h / 2.0 * (3.0 * f_curr - f_prev)
        x_next = round(x + h, 14)
        f_pred = ode_f(x_next, y_pred)
        y_corr = y + h / 2.0 * (f_pred + f_curr)

        err = abs(y_pred - y_corr) / ADAMS2_ERROR_COEFF

        if err > eps:
         
            h /= 2
            x1 = xs[-2] + h
            y1 = _rk4_step(xs[-2], ys[-2], h)
            f_prev = ode_f(xs[-2], ys[-2])
            f_curr = ode_f(x1,     y1)
            x, y   = x1, y1
            xs[-1] = x1
            ys[-1] = y1
            hs[-1] = h
            continue

        f_prev, f_curr = f_curr, ode_f(x_next, y_corr)
        x, y = x_next, y_corr
        xs.append(x)
        ys.append(y)
        hs.append(h)

        if err < C_RUNGE * eps:
            h = min(h * 2, x_end - x)

    return np.array(xs), np.array(ys), np.array(hs)


def _rk4_step(x: float, y: float, h: float) -> float:
    k1 = h * ode_f(x,           y)
    k2 = h * ode_f(x + h / 2,   y + k1 / 2)
    k3 = h * ode_f(x + h / 2,   y + k2 / 2)
    k4 = h * ode_f(x + h,       y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0


def runge_kutta4(
    x0: float = X0,
    y0: float = Y0,
    x_end: float = X_END,
    h: float = H_DEFAULT,
            ) -> tuple[np.ndarray, np.ndarray]:

    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    while x + h <= x_end + 1e-12:
        y = _rk4_step(x, y, h)
        x = round(x + h, 14)        
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def runge_error_rk4(
    x0: float = X0,
    y0: float = Y0,
    x_end: float = X_END,
    h: float = H_DEFAULT,
            ) -> tuple[np.ndarray, np.ndarray]:
   
    divisor = 2**RK4_ORDER - 1          
    xs_full, ys_h = runge_kutta4(x0, y0, x_end, h)

    errs = [0.0]                      
    x = x0
    y2 = y0
    for i in range(1, len(xs_full)):
        y2 = _rk4_step(x,       y2,     h / 2)
        y2 = _rk4_step(x + h/2, y2,     h / 2)
        x  = xs_full[i]
        errs.append(abs(ys_h[i] - y2) / divisor)

    return xs_full, np.array(errs)


def auto_step_rk4(
    x0: float = X0,
    y0: float = Y0,
    x_end: float = X_END,
    h0: float = H_DEFAULT,
    eps: float = EPS_DEFAULT,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    divisor = 2**RK4_ORDER - 1
    x, y, h = x0, y0, h0
    xs, ys, hs = [x], [y], [h]

    while x < x_end - 1e-12:
        h = min(h, x_end - x)          

        y1  = _rk4_step(x, y, h)
        y2a = _rk4_step(x,       y,  h / 2)
        y2b = _rk4_step(x + h/2, y2a, h / 2)
        err = abs(y1 - y2b) / divisor

        if err > eps:
            h /= 2
            continue                    
    
        x = x + h
        y = y2b                     
        xs.append(x)
        ys.append(y)
        hs.append(h)

        if err < C_RUNGE * eps and h * 2 <= x_end - x + 1e-12:
            h *= 2

    return np.array(xs), np.array(ys), np.array(hs)
