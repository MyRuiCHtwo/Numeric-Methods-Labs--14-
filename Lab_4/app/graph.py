from matplotlib import pyplot as plt

def step_error_dependence(h_vals, errors):
    plt.figure(figsize=(10, 6))

    # h_vals = list(h_vals)
    # errors = list(errors)
    plt.loglog(h_vals, errors, color='blue', label='Похибка R(h)')

    plt.xlabel('Крок h')
    plt.ylabel('Похибка R(h)')
    plt.title('Залежність похибки від кроку h')

    plt.grid()
    plt.legend()
    plt.show()
