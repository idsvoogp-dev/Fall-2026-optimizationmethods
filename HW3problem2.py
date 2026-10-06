import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

n = 10
k = 50
eps = 1e-6

def f(x):
    xx = np.r_[0, x, 0]
    return np.sum(x) + k/2*np.sum(np.diff(xx)**2)

def grad(x):
    xx = np.r_[0, x, 0]
    return 1 + k*(2*x - xx[:-2] - xx[2:])

# BFGS no line search
x = np.zeros(n)
H = np.eye(n)
res_no_ls = []
i = 0

while True:
    g = grad(x)
    p = -H @ g
    alpha = 1.0

    xnew = x + alpha*p

    s = xnew - x
    y = grad(xnew) - g

    if abs(y @ s) > 1e-12:
        rho = 1/(y @ s)
        I = np.eye(n)
        H = (I-rho*np.outer(s,y)) @ H @ \
            (I-rho*np.outer(y,s)) + rho*np.outer(s,s)

    x = xnew
    residual = np.linalg.norm(grad(x))
    res_no_ls.append(residual)
    i += 1

    if residual < eps:
        break

print("BFGS without line search")
print("Iterations =", i)
print("x =", x)
print("f =", f(x))
print("Residual =", residual)

# BFGS w line search
x = np.zeros(n)
H = np.eye(n)
res_ls = []
i = 0

while True:
    g = grad(x)
    p = -H @ g

    alpha = 1.0
    c = 1e-4
    beta = 0.5

    while f(x + alpha*p) > f(x) + c*alpha*(g @ p):
        alpha *= beta

    xnew = x + alpha*p

    s = xnew - x
    y = grad(xnew) - g

    if abs(y @ s) > 1e-12:
        rho = 1/(y @ s)
        I = np.eye(n)
        H = (I-rho*np.outer(s,y)) @ H @ \
            (I-rho*np.outer(y,s)) + rho*np.outer(s,s)

    x = xnew
    residual = np.linalg.norm(grad(x))
    res_ls.append(residual)
    i += 1

    if residual < eps:
        break

print("\nBFGS with line search")
print("Iterations =", i)
print("x =", x)
print("f =", f(x))
print("Residual =", residual)

# SciPy BFGS
res_scipy = []

def callback(x):
    res_scipy.append(np.linalg.norm(grad(x)))

result = minimize(
    f,
    np.zeros(n),
    jac=grad,
    method="BFGS",
    callback=callback,
    options={"gtol": eps}
)

print("\nSciPy BFGS")
print("Iterations =", result.nit)
print("x =", result.x)
print("f =", result.fun)
print("Residual =", np.linalg.norm(grad(result.x)))

#comparison
plt.figure()
plt.semilogy(res_no_ls, label="BFGS without line search")
plt.semilogy(res_ls, label="BFGS with line search")
plt.semilogy(res_scipy, label="SciPy BFGS")
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.title("BFGS Convergence")
plt.legend()
plt.grid()
plt.show()

# Cable
xx = np.arange(12)
yy = np.r_[0, result.x, 0]

plt.figure()
plt.plot(xx, yy, "o-")
plt.xlabel("Node")
plt.ylabel("Vertical Deflection")
plt.title("Optimal Cable Shape")
plt.grid()
plt.show()