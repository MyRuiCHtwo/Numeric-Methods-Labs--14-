import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, NullFormatter

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Палітра кольорів для графіків
PAL = {
    "exact"    : "#1E3A5F",
    "rk4"      : "#2563EB",
    "adams2"   : "#DC2626",
    "adams4"   : "#16A34A",
    "err_exact": "#F59E0B",
    "err_runge": "#7C3AED",
    "step"     : "#0891B2",
    "grid"     : "#E5E7EB",
    "bg"       : "#F9FAFB",
    "zero"     : "#9CA3AF",
}

LABEL = {
    "exact" : "Точний розв'язок  y = 2eˣ − x − 1",
    "rk4"   : "РК-4",
    "adams2": "Адамс-2 (прогноз+корекція)",
    "adams4": "Адамс-4 (прогноз+корекція)",
}


def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "x",
              ylabel: str = "", legend: bool = True) -> None:
    ax.set_facecolor(PAL["bg"])
    ax.grid(True, color=PAL["grid"], linewidth=0.8, zorder=0)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if legend:
        ax.legend(fontsize=9, loc="best", framealpha=0.85)


def plot_solution_comparison(
    xs_exact : np.ndarray,
    ys_exact : np.ndarray,
    xs_rk4   : np.ndarray,
    ys_rk4   : np.ndarray,
    xs_a2    : np.ndarray,
    ys_a2    : np.ndarray,
    h        : float,
    show     : bool = True,
) -> plt.Figure:
  
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PAL["bg"])

    ax.plot(xs_exact, ys_exact, color=PAL["exact"],  linewidth=2.5,
            label=LABEL["exact"], zorder=5)
    ax.plot(xs_rk4,   ys_rk4,   color=PAL["rk4"],    linewidth=1.8,
            linestyle="--", marker="o", markersize=3, label=LABEL["rk4"],    zorder=4)
    ax.plot(xs_a2,    ys_a2,    color=PAL["adams2"],  linewidth=1.8,
            linestyle="-.", marker="s", markersize=3, label=LABEL["adams2"], zorder=3)
    _style_ax(ax,
              title=f"Розв'язок задачі Коші   y' = x + y,   y(0) = 1   (h = {h})",
              ylabel="y(x)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_exact_error_adams2(
    xs   : np.ndarray,
    errs : np.ndarray,
    h    : float,
    show : bool = True,
) -> plt.Figure:
   
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PAL["bg"])
    ax.semilogy(xs, np.where(errs < 1e-16, 1e-16, errs),
                color=PAL["err_exact"], linewidth=2.0,
                marker="o", markersize=4, label=f"|y_n − y(x_n)|")
    _style_ax(ax,
              title=f"окальна похибка — Адамс-2   (h = {h})",
              ylabel="похибка (log)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_runge_error_adams2(
    xs    : np.ndarray,
    errs  : np.ndarray,
    eps   : float,
    h     : float,
    show  : bool = True,
) -> plt.Figure:
   
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PAL["bg"])

    safe_errs = np.where(errs < 1e-16, 1e-16, errs)
    ax.semilogy(xs, safe_errs, color=PAL["err_runge"], linewidth=2.0,
                marker="s", markersize=4, label="Оцінка Рунге  |y_pred − y_corr| / 3")
    ax.axhline(eps, color=PAL["adams2"], linewidth=1.5,
               linestyle="--", label=f"ε = {eps:.1e}")
    ax.axhline(0.5 * eps, color=PAL["rk4"], linewidth=1.0,
               linestyle=":", label=f"0.5·ε = {0.5*eps:.1e}")

    _style_ax(ax,
              title=f"Оцінка похибки по Рунге — Адамс-2   (h = {h})",
              ylabel="похибка (log)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig
 

def plot_step_adams2(
    xs : np.ndarray,
    hs : np.ndarray,
    eps: float,
    show: bool = True,
) -> plt.Figure:
  
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PAL["bg"])
    ax.step(xs, hs, where="post", color=PAL["step"], linewidth=2.2,
            label="h(x) — автовибір кроку")
    _style_ax(ax,
              title=f"Залежність кроку h(x) — Адамс-2   (ε = {eps:.1e})",
              ylabel="крок  h")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_exact_error_rk4(
    results: list[tuple[np.ndarray, np.ndarray, float]],
    show   : bool = True,
            ) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PAL["bg"])

    colors = ["#1E3A5F", "#2563EB", "#7C3AED", "#DC2626", "#F59E0B"]
    for i, (xs, errs, h) in enumerate(results):
        safe = np.where(errs < 1e-16, 1e-16, errs)
        ax.semilogy(xs, safe, color=colors[i % len(colors)],
                    linewidth=1.8, marker="o", markersize=3,
                    label=f"h = {h:.3f}")

    _style_ax(ax,
              title="Точна локальна похибка РК-4 для різних кроків h",
              ylabel="похибка (log)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_runge_error_rk4(
    xs    : np.ndarray,
    errs  : np.ndarray,
    eps   : float,
    h     : float,
    show  : bool = True,
            ) -> plt.Figure:
  
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PAL["bg"])

    safe = np.where(errs < 1e-16, 1e-16, errs)
    ax.semilogy(xs, safe, color=PAL["err_runge"], linewidth=2.0,
                marker="D", markersize=4,
                label="Оцінка Рунге  |y_h − y_{h/2}| / 15")
    ax.axhline(eps, color=PAL["adams2"], linewidth=1.5,
               linestyle="--", label=f"ε = {eps:.1e}")
    ax.axhline(0.5 * eps, color=PAL["rk4"], linewidth=1.0,
               linestyle=":", label=f"0.5·ε = {0.5*eps:.1e}")

    _style_ax(ax,
              title=f"Оцінка похибки по Рунге — РК-4   (h = {h})",
              ylabel="похибка (log)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_step_rk4(
    xs  : np.ndarray,
    hs  : np.ndarray,
    eps : float,
    show: bool = True,
            ) -> plt.Figure:
   
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor(PAL["bg"])
    ax.step(xs, hs, where="post", color=PAL["rk4"], linewidth=2.2,
            label="h(x) — автовибір кроку")
    _style_ax(ax,
              title=f"Залежність кроку h(x) — РК-4   (ε = {eps:.1e})",
              ylabel="крок  h")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_convergence_table(
    rows  : list[dict],
    method: str = "РК-4",
    order : int  = 4,
    show  : bool = True,
            ) -> plt.Figure:
   
    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    hs     = [r["h"]     for r in rows if r["ratio"] is not None]
    labels = [f"h={h:.4f}" for h in hs]

    if not ratios:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(PAL["bg"])
    bars = ax.bar(labels, ratios, color=PAL["rk4"], edgecolor="white", linewidth=0.8)
    ax.axhline(2 ** order, color=PAL["adams2"], linewidth=1.8,
               linestyle="--", label=f"Теоретичне значення = 2^{order} = {2**order}")
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    _style_ax(ax,
              title=f"Підтвердження порядку збіжності — {method}",
              xlabel="крок h", ylabel="err(2h) / err(h)")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_dashboard(
    xs_exact: np.ndarray, ys_exact: np.ndarray,
    xs_rk4  : np.ndarray, ys_rk4  : np.ndarray,
    xs_a2   : np.ndarray, ys_a2   : np.ndarray,
    errs_exact_rk4   : np.ndarray,
    errs_runge_rk4   : np.ndarray,
    errs_exact_a2    : np.ndarray,
    errs_runge_a2    : np.ndarray,
    xs_auto_rk4 : np.ndarray, hs_auto_rk4 : np.ndarray,
    xs_auto_a2  : np.ndarray, hs_auto_a2  : np.ndarray,
    h   : float,
    eps : float,
    show: bool = True,
        ) -> plt.Figure:
   
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(PAL["bg"])
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── [0,0] Розв'язки ───────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_facecolor(PAL["bg"])
    ax00.grid(True, color=PAL["grid"], linewidth=0.7)
    ax00.plot(xs_exact, ys_exact, color=PAL["exact"],  lw=2.2, label="Точний")
    ax00.plot(xs_rk4,   ys_rk4,   color=PAL["rk4"],    lw=1.6,
              ls="--", marker="o", ms=2.5, label="РК-4")
    ax00.plot(xs_a2,    ys_a2,    color=PAL["adams2"],  lw=1.6,
              ls="-.", marker="s", ms=2.5, label="Адамс-2")
 
    ax00.set_title(f"Розв'язок   (h = {h})", fontsize=11, fontweight="bold")
    ax00.legend(fontsize=8)

    # ── [0,1] Точна похибка ────────────────────────────────────────────
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.set_facecolor(PAL["bg"])
    ax01.grid(True, color=PAL["grid"], linewidth=0.7)
    safe_rk4 = np.where(errs_exact_rk4 < 1e-16, 1e-16, errs_exact_rk4)
    safe_a2  = np.where(errs_exact_a2  < 1e-16, 1e-16, errs_exact_a2)
    ax01.semilogy(xs_rk4,  safe_rk4, color=PAL["rk4"],   lw=1.8,
                  marker="o", ms=2.5, label="РК-4")
    ax01.semilogy(xs_a2,   safe_a2,  color=PAL["adams2"], lw=1.8,
                  marker="s", ms=2.5, label="Адамс-2")
    ax01.set_title("Точна локальна похибка", fontsize=11, fontweight="bold")
    ax01.set_ylabel("похибка (log)", fontsize=10)
    ax01.legend(fontsize=8)

    # ── [1,0] Похибка Рунге ────────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.set_facecolor(PAL["bg"])
    ax10.grid(True, color=PAL["grid"], linewidth=0.7)
    safe_r_rk4 = np.where(errs_runge_rk4 < 1e-16, 1e-16, errs_runge_rk4)
    safe_r_a2  = np.where(errs_runge_a2  < 1e-16, 1e-16, errs_runge_a2)
    ax10.semilogy(xs_rk4,  safe_r_rk4, color=PAL["rk4"],    lw=1.8,
                  marker="D", ms=2.5, label="Рунге РК-4")
    ax10.semilogy(xs_a2,   safe_r_a2,  color=PAL["adams2"],  lw=1.8,
                  marker="s", ms=2.5, label="Рунге Адамс-2")
    ax10.axhline(eps, color="#DC2626", lw=1.3, ls="--", label=f"ε={eps:.0e}")
    ax10.set_title("Оцінка похибки по Рунге", fontsize=11, fontweight="bold")
    ax10.set_ylabel("похибка (log)", fontsize=10)
    ax10.legend(fontsize=8)

    # ── [1,1] Обидва автокроки разом ──────────────────────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.set_facecolor(PAL["bg"])
    ax11.grid(True, color=PAL["grid"], linewidth=0.7)
    ax11.step(xs_auto_rk4, hs_auto_rk4, where="post",
              color=PAL["rk4"],   lw=2.0, label="РК-4 автокрок")
    ax11.step(xs_auto_a2,  hs_auto_a2,  where="post",
              color=PAL["adams2"], lw=2.0, ls="--", label="Адамс-2 автокрок")
    ax11.set_title(f"h(x) — автовибір кроку  (ε={eps:.0e})", fontsize=11, fontweight="bold")
    ax11.set_ylabel("крок h", fontsize=10)
    ax11.legend(fontsize=8)

    # ── [2,0] Автокрок РК-4 детально ──────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    ax20.set_facecolor(PAL["bg"])
    ax20.grid(True, color=PAL["grid"], linewidth=0.7)
    ax20.step(xs_auto_rk4, hs_auto_rk4, where="post",
              color=PAL["rk4"], lw=2.2, label="h(x)")
    ax20.set_title("Автовибір кроку — РК-4", fontsize=11, fontweight="bold")
    ax20.set_xlabel("x", fontsize=10)
    ax20.set_ylabel("крок h", fontsize=10)
    ax20.legend(fontsize=8)

    # ── [2,1] Автокрок Адамс-2 детально ──────────────────────────────
    ax21 = fig.add_subplot(gs[2, 1])
    ax21.set_facecolor(PAL["bg"])
    ax21.grid(True, color=PAL["grid"], linewidth=0.7)
    ax21.step(xs_auto_a2, hs_auto_a2, where="post",
              color=PAL["adams2"], lw=2.2, ls="--", label="h(x)")
    ax21.set_title("Автовибір кроку — Адамс-2", fontsize=11, fontweight="bold")
    ax21.set_xlabel("x", fontsize=10)
    ax21.set_ylabel("крок h", fontsize=10)
    ax21.legend(fontsize=8)

    fig.suptitle(
        "Лабораторна робота №10  |  y′ = x + y,   y(0) = 1,   x ∈ [0, 1]",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig
