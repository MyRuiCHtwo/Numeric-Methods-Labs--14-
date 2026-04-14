import numpy as np

from app.functions.funcs import f, sinpson_method, N_error_dependence, adaptive_simpson
from app.graphs import show_N_error_dependence, show_server_load_function



def main():
    a, b = 0, 24
    N = 1000

    I0, N_values, errors, n_opt, eps_opt = N_error_dependence(a, b)
    I, x, y = sinpson_method(f, a, b, N)

    print(f"  Точне значення інтегралу: {I0:.15f}\n")
    print(f"  Приблизне значення інтегралу: {I:.15f}\n")
    print(f"  Оптимальне N: {n_opt},\n  Похибка при оптимальному N: {eps_opt:.2e}\n")


    show_server_load_function(x, y)
    show_N_error_dependence(N_values, errors, n_opt, eps_opt)

    N0 = 16
    I_n0, _, _ = sinpson_method(f, a, b, N0)
    I_n0_2, _, _ = sinpson_method(f, a, b, (N0 // 2))
    I_n0_4, _, _ = sinpson_method(f, a, b, (N0 // 4))

    rat = (I_n0_4 - I_n0_2) / (I_n0_2 - I_n0)
    p = np.log2(abs(rat))

    Ir = I_n0 + (I_n0 - I_n0_2) / 15
    IA = I_n0 - ((I_n0 - I_n0_2)**2) / (I_n0 - 2*I_n0_2 + I_n0_4)


    err0 = abs(I_n0 - I0)
    epsR = abs(Ir - I0)
    epsA = abs(IA - I0)

    print(f"  При N = {N0},\n  Похибка: {err0:.2e}\n")
    print(f"  Уточнене Ir: {Ir:.15f}")
    print(f"  Похибка за Рунге-Ромбергом: {epsR:.2e}\n ")
    print(f"  Обчислений порядок точності p: {p:.4f}")
    print(f"  Уточнене значення за Ейткеном: {IA:.15f}")
    print(f"  Похибка методу Ейткена: {epsA:.2e}\n")

    I_ad = adaptive_simpson(f, a, b, 1e-12)
    print(f"  Результат адаптивного методу Сімпсона: {I_ad:.15f}")
    





if __name__ == "__main__":
    main()

    