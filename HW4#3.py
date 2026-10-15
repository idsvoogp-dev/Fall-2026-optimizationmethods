
import numpy as np

def simplex(c, A, b, tol=1e-9):
    m, n = A.shape

    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:m, :n] = A
    tableau[:m, n:n + m] = np.eye(m)
    tableau[:m, -1] = b
    tableau[-1, :n] = -c

    basis = list(range(n, n + m))
    iterations = 0

    while np.min(tableau[-1, :-1]) < -tol:
        entering = np.argmin(tableau[-1, :-1])

        eligible = [
            i for i in range(m)
            if tableau[i, entering] > tol
        ]

        if not eligible:
            raise ValueError("The problem is unbounded.")

        ratios = [
            tableau[i, -1] / tableau[i, entering]
            for i in eligible
        ]
        leaving_row = eligible[np.argmin(ratios)]

        pivot = tableau[leaving_row, entering]
        tableau[leaving_row] /= pivot

        for i in range(m + 1):
            if i != leaving_row:
                tableau[i] -= (
                    tableau[i, entering] *
                    tableau[leaving_row]
                )

        basis[leaving_row] = entering
        iterations += 1

    solution = np.zeros(n + m)

    for i, variable in enumerate(basis):
        solution[variable] = tableau[i, -1]

    return solution[:n], tableau[-1, -1], iterations


c = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0])

A = np.array([
    [10.0, 4.0, 16.0, 7.5, 5.6, 4.4],
    [2.0, 4.0, 1.6, 2.4, 2.8, 3.3]
])

b = np.array([400.0, 120.0])

x, profit, iterations = simplex(c, A, b)

for i, value in enumerate(x, start=1):
    print(f"x{i} = {value:.4f}")

print(f"Maximum profit = ${profit:.2f}")
print(f"Simplex iterations = {iterations}")
