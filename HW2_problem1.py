import numpy as np


eps = 1e-6

def f(z):
    x, y = z
    return 5*x**2 - 5*x*y + 2.5*y**2 - x - 1.5*y

def grad(z):
    x, y = z
    return np.array([10*x - 5*y - 1, -5*x + 5*y - 1.5])

H = np.array([[10., -5.],
              [-5., 5.]])

# Optimal Steepest Descent
z = np.array([0., 0.])
i = 0

while True:
    g = grad(z)
    d = -g
    alpha = (g @ g) / (g @ H @ g)
    z = z + alpha*d
    residual = np.linalg.norm(grad(z))
    i += 1

    if residual < eps:
        break

print("Iterations =", i)
print("x =", z[0])
print("y =", z[1])
print("f(x,y) =", f(z))
print("Residual =", residual)