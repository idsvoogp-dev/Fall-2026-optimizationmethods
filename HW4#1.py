
import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, 4, 400)

# Constraint lines
y_potato = np.full_like(x1, 3)
y_carrot = np.full_like(x1, 4)
y_fertilizer = 5 - x1

# boundary
x_feasible = [0, 3, 3, 1, 0]
y_feasible = [0, 0, 2, 4, 4]

plt.figure(figsize=(8, 6))

plt.plot(x1, y_potato, label=r'$x_1 \leq 3$')
plt.plot(x1, y_carrot, label=r'$x_2 \leq 4$')
plt.plot(x1, y_fertilizer, label=r'$x_1+x_2 \leq 5$')

plt.fill(x_feasible, y_feasible, alpha=0.25,
         label='Feasible region')

plt.scatter(1, 4, color='red', zorder=5)
plt.annotate('Optimal point (1, 4)', (1, 4),
             xytext=(1.25, 3.5),
             arrowprops=dict(arrowstyle='->'))

plt.xlim(0, 4)
plt.ylim(0, 5)
plt.xlabel('Potatoes, $x_1$ (tons)')
plt.ylabel('Carrots, $x_2$ (tons)')
plt.title('Farmer’s Plan: Graphical Method')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


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


c = np.array([1200.0, 1700.0])

A = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

b = np.array([3.0, 4.0, 5.0])

x, profit, iterations = simplex(c, A, b)

print("Potatoes (tons):", x[0])
print("Carrots (tons):", x[1])
print("Maximum profit ($):", profit)
print("Simplex iterations:", iterations)