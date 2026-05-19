import numpy as np

from app.math_core import (
    X0, Y0, X_END,
    exact,
    runge_kutta4,
    adams2,
    auto_step_rk4,
    auto_step_adams2,
    runge_error_rk4,
)
from app.errors import (
    local_exact_error,
    runge_error_estimate_rk4,
    runge_error_estimate_adams2,
    check_step_optimality,
    convergence_table,
)
from app.ui_output import (
    plot_solution_comparison,
    plot_exact_error_adams2,
    plot_runge_error_adams2,
    plot_step_adams2,
    plot_exact_error_rk4,
    plot_runge_error_rk4,
    plot_step_rk4,
    plot_convergence_table,
    plot_dashboard,
)



H          = 0.1         
EPS        = 1e-4         
H0_AUTO    = 0.1         
SHOW_PLOTS = True         

H_LIST = [0.2, 0.1, 0.05, 0.025]

SEP  = "=" * 72
SEP2 = "-" * 52


def _header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def _print_solution_table(
    label: str,
    xs   : np.ndarray,
    ys   : np.ndarray,
    errs : np.ndarray,
) -> None:
    print(f"\n  {'x':>8}  {'y_числ':>14}  {'y_точне':>14}  {'|похибка|':>12}")
    print("  " + SEP2)
    for x, y, e in zip(xs, ys, errs):
        print(f"  {x:>8.4f}  {y:>14.8f}  {exact(x):>14.8f}  {e:>12.4e}")


def _print_optimality(label: str, report: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    Максимальна похибка : {report['max_err']:.4e}")
    print(f"    Середня похибка     : {report['mean_err']:.4e}")
    print(f"    Вузлів з err > ε    : {report['n_exceeded']}")
    print(f"    Вузлів з err < ε/2  : {report['n_small']}")
    print(f"    Висновок            : {report['verdict']}")


def _print_convergence_table(rows: list[dict], method: str) -> None:
    print(f"\n  {'h':>10}  {'max|err|':>14}  {'err(2h)/err(h)':>16}  {'очікув.':>8}")
    print("  " + SEP2)
    for r in rows:
        ratio_str = f"{r['ratio']:.4f}" if r["ratio"] is not None else "     —"
        print(f"  {r['h']:>10.4f}  {r['max_err']:>14.4e}"
              f"  {ratio_str:>16}  {r['expected']:>8}")


def _print_step_stats(label: str, hs: np.ndarray) -> None:
    print(f"\n  [{label}]  кроків={len(hs)}   "
          f"h_min={hs.min():.4f}   h_max={hs.max():.4f}   "
          f"h_mean={hs.mean():.4f}")


def main() -> None:

    # Ч.1 

    xs_exact_fine = np.linspace(X0, X_END, 500)
    ys_exact_fine = exact(xs_exact_fine)
      
    _header("Аналітичний розв'язок")
    print(f"\n  y' = x + y,   y(0) = {Y0},   x ∈ [{X0}, {X_END}]")
    print(f"  Точний розв'язок: y(x) = 2·eˣ − x − 1")
    print(f"\n  {'x':>8}  {'y_точне':>14}")
    for x in np.arange(X0, X_END + 1e-12, H):
        print(f"  {x:>8.4f}  {exact(x):>14.8f}")

    #  Адамс_2 зі сталим кроком
    _header("Адамс-2  (прогноз + корекція,  сталий крок)")
    xs_a2, ys_a2 = adams2(X0, Y0, X_END, H)
    errs_exact_a2 = local_exact_error(xs_a2, ys_a2)
    _print_solution_table("Адамс-2", xs_a2, ys_a2, errs_exact_a2)

    # точна локальна похибка Адамс_2
    _header("Точна локальна похибка — Адамс-2")
    print(f"  Максимальна похибка: {errs_exact_a2.max():.4e}")
    plot_exact_error_adams2(xs_a2, errs_exact_a2, H, show=SHOW_PLOTS)

    # оцінка похибки по Рунге + перевірка оптимальності кроку
    _header("Оцінка похибки по Рунге — Адамс-2")
    xs_re_a2, errs_runge_a2 = runge_error_estimate_adams2(X0, Y0, X_END, H)
    opt_a2 = check_step_optimality(errs_runge_a2, EPS)
    _print_optimality("Адамс-2  Рунге", opt_a2)
    plot_runge_error_adams2(xs_re_a2, errs_runge_a2, EPS, H, show=SHOW_PLOTS)

    # автовибір кроку для Фдамс_2
    _header("Автовибір кроку — Адамс-2")
    xs_auto_a2, ys_auto_a2, hs_auto_a2 = auto_step_adams2(
        X0, Y0, X_END, H0_AUTO, EPS
    )
    _print_step_stats("Адамс-2 автокрок", hs_auto_a2)
    plot_step_adams2(xs_auto_a2, hs_auto_a2, EPS, show=SHOW_PLOTS)
  

    # Ч.2

    # РК-4 зі сталим кроком
    _header("Рунге-Кутта-4  (сталий крок)")
    xs_rk4, ys_rk4 = runge_kutta4(X0, Y0, X_END, H)
    errs_exact_rk4 = local_exact_error(xs_rk4, ys_rk4)
    _print_solution_table("РК-4", xs_rk4, ys_rk4, errs_exact_rk4)

    # точна похибка РК-4 для різних h
    _header("Залежність точної похибки від кроку h — РК-4")
    rk4_err_results = []
    max_err_list    = []
    for h_i in H_LIST:
        xs_i, ys_i = runge_kutta4(X0, Y0, X_END, h_i)
        errs_i = local_exact_error(xs_i, ys_i)
        max_err_list.append(float(errs_i.max()))
        rk4_err_results.append((xs_i, errs_i, h_i))
        print(f"  h = {h_i:.3f}   max|err| = {errs_i.max():.4e}")
    plot_exact_error_rk4(rk4_err_results, show=SHOW_PLOTS)

    # оцінка похибки по Рунге для РК-4 + оптимальність кроку
    _header("Оцінка похибки по Рунге — РК-4")
    xs_runge_rk4, errs_runge_rk4 = runge_error_estimate_rk4(X0, Y0, X_END, H)
    opt_rk4 = check_step_optimality(errs_runge_rk4, EPS)
    _print_optimality("РК-4  Рунге", opt_rk4)

    # Необхідний крок для заданої точності
    max_runge = errs_runge_rk4.max()
    if max_runge > 0:
        h_needed = H * (EPS / max_runge) ** (1.0 / 4)
        print(f"\n  Оцінка необхідного кроку для ε = {EPS:.0e}: h* ≈ {h_needed:.5f}")

    plot_runge_error_rk4(xs_runge_rk4, errs_runge_rk4, EPS, H, show=SHOW_PLOTS)

    # Таблиця підтвердження порядку збіжності
    _header("Таблиця підтвердження порядку збіжності — РК-4")
    conv_rows = convergence_table(H_LIST, max_err_list, order=4)
    _print_convergence_table(conv_rows, "РК-4")
    plot_convergence_table(conv_rows, method="РК-4", order=4, show=SHOW_PLOTS)

    # автовибір кроку РК-4
    _header("Автовибір кроку — РК-4")
    xs_auto_rk4, ys_auto_rk4, hs_auto_rk4 = auto_step_rk4(
        X0, Y0, X_END, H0_AUTO, EPS
    )
    _print_step_stats("РК-4 автокрок", hs_auto_rk4)
    plot_step_rk4(xs_auto_rk4, hs_auto_rk4, EPS, show=SHOW_PLOTS)

  
    _header("Зведений дашборд")
    plot_solution_comparison(
        xs_exact_fine, ys_exact_fine,
        xs_rk4,        ys_rk4,
        xs_a2,         ys_a2,
        H,             show=SHOW_PLOTS,
    )

    # plot_dashboard(
    #     xs_exact_fine,  ys_exact_fine,
    #     xs_rk4,         ys_rk4,
    #     xs_a2,          ys_a2,
    #     errs_exact_rk4,
    #     errs_runge_rk4,
    #     errs_exact_a2,
    #     errs_runge_a2,
    #     xs_auto_rk4,    hs_auto_rk4,
    #     xs_auto_a2,     hs_auto_a2,
    #     H, EPS,
    #     show=SHOW_PLOTS,
    # )

    _header("Виконання завершено")
   

if __name__ == "__main__":
    main()
