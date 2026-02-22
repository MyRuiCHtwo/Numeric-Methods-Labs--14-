import numpy as np



def get_params_for_plot(N):
    x, y = np.zeros(N + 1), np.zeros(N + 1)
    xm, ym = np.zeros(20*N + 1), np.zeros(20*N + 1)
    s = np.zeros(20*N + 1)
    eps = np.zeros(20*N + 1)

    with open("Lab_1/data/input.txt", 'r', encoding='utf-8') as fh:
        for i in range(N + 1):
                line = fh.readline().split('|')
                x[i] = float(line[1])
                y[i] = float(line[2])

    with open("Lab_1/data/output.txt", 'r', encoding='utf-8') as fh:
        lines = fh.readlines() 
        for i in range(len(lines)):
            parts = lines[i].split('|')
     
            xm[i] = float(parts[1])
            ym[i] = float(parts[2])
            s[i] = float(parts[3])
            eps[i] = float(parts[4])

            
    return x, y, xm, ym, s, eps


    