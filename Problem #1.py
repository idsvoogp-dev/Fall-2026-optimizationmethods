import numpy as np
import matplotlib.pyplot as plt
import math

eps = 1e-6
def f(x):
    return -1.5*x**6 - 2*x**4 + 12*x

def df(x):
    return -9*x**5 - 8*x**3 + 12

def d2f(x):
    return -45*x**4 - 24*x**2

# Part (a): Plot
x = np.linspace(-2, 2, 500)
plt.plot(x, f(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

#golden section
xl, xu = 0, 2
R = (math.sqrt(5) - 1)/2
i = 0

while True:
    x1 = xu - R*(xu - xl)
    x2 = xl + R*(xu - xl)

    if f(x1) < f(x2):
        xl = x1
    else:
        xu = x2

    xopt = (xl + xu)/2
    residual = (xu - xl)/max(abs(xopt), 1)
    i += 1

    if residual < eps:
        break

print("\nGolden Section:")
print("Iterations =", i)
print("x =", xopt)
print("f(x) =", f(xopt))
print("Residual =", residual)

#Newton method
x = 2
i = 0

while True:
    xnew = x - df(x)/d2f(x)
    residual = abs(df(xnew))
    i += 1

    if residual < eps:
        break

    x = xnew

print("\nNewton's Method:")
print("Iterations =", i)
print("x =", xnew)
print("f(x) =", f(xnew))
print("Residual =", residual)