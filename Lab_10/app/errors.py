import numpy as np
from app.math_core import exact, runge_error_rk4, ode_f, C_RUNGE, _rk4_step


def local_exact_error(
    xs: np.ndarray,
    ys: np.ndarray,
            ) -> np.ndarray:
  
    y_ex = exact(xs)
    return np.abs(ys - y_ex)


def runge_error_estimate_rk4(
    x0: float,
    y0: float,
    x_end: float,
    h: float,
            ) -> tuple[np.ndarray, np.ndarray]:
  
    return runge_error_rk4(x0, y0, x_end, h)


def runge_error_estimate_adams2(
    x0: float,
    y0: float,
    x_end: float,
    h: float,
            ) -> tuple[np.ndarray, np.ndarray]:
   
    COEFF = 3.0

    x1 = x0 + h
    y1 = _rk4_step(x0, y0, h)

    xs   = [x0, x1]
    errs = [0.0, 0.0]          

    f_prev = ode_f(x0, y0)
    f_curr = ode_f(x1, y1)
    x, y   = x1, y1

    while x + h <= x_end + 1e-12:
        y_pred = y + h / 2.0 * (3.0 * f_curr - f_prev)
        x_next = round(x + h, 14)
        f_pred = ode_f(x_next, y_pred)
        y_corr = y + h / 2.0 * (f_pred + f_curr)

        err = abs(y_pred - y_corr) / COEFF
        errs.append(err)
        xs.append(x_next)

        f_prev, f_curr = f_curr, ode_f(x_next, y_corr)
        x, y = x_next, y_corr

    return np.array(xs), np.array(errs)


def check_step_optimality(
    errs: np.ndarray,
    eps: float,
            ) -> dict:
    
    max_err  = float(np.max(errs))
    mean_err = float(np.mean(errs))
    n_exc    = int(np.sum(errs > eps))
    n_small  = int(np.sum(errs < C_RUNGE * eps))
    n_total  = len(errs)

    if n_exc > 0:
        status  = "too_large"
        verdict = (
            f"Крок ЗАВЕЛИКИЙ: {n_exc}/{n_total} вузлів мають err > ε = {eps:.1e}. "
            f"Рекомендується зменшити крок."
        )
    elif n_small == n_total:
        status  = "too_small"
        verdict = (
            f"Крок ЗАМАЛИЙ: всі {n_total} вузлів мають err < {C_RUNGE}·ε = {C_RUNGE*eps:.1e}. "
            f"Крок можна збільшити вдвоє."
        )
    else:
        status  = "optimal"
        verdict = (
            f"Крок ОПТИМАЛЬНИЙ: максимальна похибка {max_err:.2e} "
            f"відповідає заданій точності ε = {eps:.1e}."
        )

    return {
        "status"     : status,
        "max_err"    : max_err,
        "mean_err"   : mean_err,
        "n_exceeded" : n_exc,
        "n_small"    : n_small,
        "verdict"    : verdict,
    }


def convergence_table(
    h_list:   list[float],
    err_list: list[float],
    order:    int,
            ) -> list[dict]:
   
    rows = []
    for i, (h, e) in enumerate(zip(h_list, err_list)):
        ratio = err_list[i - 1] / e if i > 0 and abs(e) > 1e-16 else None
        rows.append({
            "h"        : h,
            "max_err"  : e,
            "ratio"    : ratio,
            "expected" : 2 ** order,
        })
    return rows
