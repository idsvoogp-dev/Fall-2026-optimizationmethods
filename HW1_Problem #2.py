import numpy as np
import matplotlib.pyplot as plt

eps = 1e-6
delta = 0.01

def f(x):
    return 3 + 6*x + 5*x**2 + 3*x**3 + 4*x**4

def df(x):
    return 6 + 10*x + 9*x**2 + 16*x**3

def d2f(x):
    return 10 + 18*x + 48*x**2

x = np.linspace(-1.5, 1, 500)
plt.plot(x, f(x))
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid()
plt.show()

#b
x = -1
res1 = []
i = 0

while True:
    x = x - df(x)/d2f(x)
    r = abs(df(x))
    res1.append(r)
    i += 1

    if r < eps:
        break

print("Analytical Newton:")
print("Iterations =", i)
print("x =", x)
print("f(x) =", f(x))
print("Residual =", r)

# finite
x = -1
res2 = []
i = 0

while True:
    h = delta*abs(x)

    d1 = (f(x+h) - f(x-h))/(2*h)
    d2 = (f(x+h) - 2*f(x) + f(x-h))/h**2

    x = x - d1/d2

    h = delta*abs(x)
    r = abs((f(x+h) - f(x-h))/(2*h))
    res2.append(r)
    i += 1

    if r < eps:
        break

print("\nFinite Difference Newton:")
print("Iterations =", i)
print("x =", x)
print("f(x) =", f(x))
print("Residual =", r)