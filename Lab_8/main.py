import numpy as np

import os

import app.additional as add
import app.methods as meth
import app.UI_output as ui


def main():
   
    eps = 1e-6               
    a, b = -3.0, 3.0        
    h = 0.1                 
               
    tab_file = "Lab_8/data/tabulated_func.txt"
    coef_file = "Lab_8/data/Coefs.txt"

    f = add.f
    df = add.df
    d2f = add.d2f
      
    tab_data = add.tabulate_function(a, b, tab_file, h)
    root_inc_approx, root_dec_approx = add.find_initial_roots(tab_data)
    
    if root_inc_approx is None or root_dec_approx is None:
        print("Помилка: Не вдалося автоматично знайти початкові наближення для обох ділянок.")
        return

 
    ui.print_transcendental_header("зростання")
    
    r_si_inc, i_si_inc = meth.method_simple_iteration(f, root_inc_approx, eps)
    ui.print_transcendental_row("Простої ітерації", r_si_inc, i_si_inc)
    
    r_n_inc, i_n_inc = meth.method_newton(f, df, root_inc_approx, eps)
    ui.print_transcendental_row("Ньютона (дотичних)", r_n_inc, i_n_inc)
    
    r_ch_inc, i_ch_inc = meth.method_chebyshev(f, df, d2f, root_inc_approx, eps)
    ui.print_transcendental_row("Чебишева (3-го порядку)", r_ch_inc, i_ch_inc)
    
    r_chd_inc, i_chd_inc = meth.method_chord(f, root_inc_approx, eps)
    ui.print_transcendental_row("Хорд (січних)", r_chd_inc, i_chd_inc)
    
    r_par_inc, i_par_inc = meth.method_parabola(f, root_inc_approx, eps)
    real_r_par_inc = r_par_inc.real if isinstance(r_par_inc, complex) else r_par_inc
    ui.print_transcendental_row("Парабол (Мюллера)", real_r_par_inc, i_par_inc)
    
    r_inv_inc, i_inv_inc = meth.method_inverse_interpolation(f, root_inc_approx, eps)
    ui.print_transcendental_row("Зворотної інтерполяції", r_inv_inc, i_inv_inc)

   
    ui.print_transcendental_header("спадання")
    
    r_si_dec, i_si_dec = meth.method_simple_iteration(f, root_dec_approx, eps)
    ui.print_transcendental_row("Простої ітерації", r_si_dec, i_si_dec)
    
    r_n_dec, i_n_dec = meth.method_newton(f, df, root_dec_approx, eps)
    ui.print_transcendental_row("Ньютона (дотичних)", r_n_dec, i_n_dec)
    
    r_ch_dec, i_ch_dec = meth.method_chebyshev(f, df, d2f, root_dec_approx, eps)
    ui.print_transcendental_row("Чебишева (3-го порядку)", r_ch_dec, i_ch_dec)
    
    r_chd_dec, i_chd_dec = meth.method_chord(f, root_dec_approx, eps)
    ui.print_transcendental_row("Хорд (січних)", r_chd_dec, i_chd_dec)
    
    r_par_dec, i_par_dec = meth.method_parabola(f, root_dec_approx, eps)
    real_r_par_dec = r_par_dec.real if isinstance(r_par_dec, complex) else r_par_dec
    ui.print_transcendental_row("Парабол (Мюллера)", real_r_par_dec, i_par_dec)
    
    r_inv_dec, i_inv_dec = meth.method_inverse_interpolation(f, root_dec_approx, eps)
    ui.print_transcendental_row("Зворотної інтерпоняції", r_inv_dec, i_inv_dec)
    print("-"*75)

   
    try:      
        coeffs = add.read_polynomial_coefficients(coef_file)
                
        poly_x0 = 1.0 
        
        n_root, n_iter = meth.method_newton_horner(coeffs, poly_x0, eps)
        
        lin_roots, lin_iter = meth.method_lin(coeffs, eps)
              
        ui.print_polynomial_results(n_root, n_iter, lin_roots, lin_iter)
        
    except FileNotFoundError as e:
        print(f"\nНе знайдено файл вхідних даних: {e}")
        return
    except ValueError as e:
        print(f"\nМатематична помилка обчислень: {e}")
        return

    
    ui.plot_transcendental_function(
        f, a, b, 
        roots=[r_n_inc, r_n_dec], 
        root_labels=["Зростання", "Спадання"]
    )
       
    ui.plot_polynomial_function(
        coeffs, x_start=-1.0, x_end=4.0, 
        real_root=n_root, 
        complex_roots=lin_roots
    )
    print("\nЗавершено.")  


if __name__ == "__main__":
    main()
