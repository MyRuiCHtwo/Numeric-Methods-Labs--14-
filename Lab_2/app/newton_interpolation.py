
def get_newton_polynomial(x_target, x, coef_n):
    n = len(x)
    result = coef_n[0]
    product = 1.0

    for i in range(1, n):
        product *= (x_target - x[i-1])
        result += coef_n[i] * product
    
    return result
    