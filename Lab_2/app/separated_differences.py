import csv


def get_separated_differences(x, y):
    n = len(x)
    separated_differences = [[0] * n for _ in range(n)]

    for i in range(n):
        separated_differences[i][0] = y[i]
    
    for j in range(1, n):
        for i in range(n - j):
            separated_differences[i][j] = (separated_differences[i + 1][j - 1] - separated_differences[i][j - 1]) / (x[i + j] - x[i])
    
    try:
        with open('Lab_2/data/separated_differences.csv', 'w', encoding='utf-8') as fh:
       
            headers = [f"Order_{k}" for k in range(n)]
            fh.write(",".join(headers) + "\n")
            
            for row in separated_differences:
                fh.write(",".join(map(str, row)) + "\n")
    except FileNotFoundError:
        print("Помилка: Перевірте наявність папки для запису файлу.")


    coefs_for_newton = separated_differences[0][:n]

    return coefs_for_newton


