import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

n = 20
alpha = 0.15
beta = 0.05
gamma = 2.0
eps = 1e-6

def f(x):
    i = np.arange(1, n+1)
    return np.sum(np.exp(-alpha*i*x) + beta*x**2) + gamma*np.sum(np.diff(x)**2)

def grad(x):
    i = np.arange(1, n+1)
    g = -alpha*i*np.exp(-alpha*i*x) + 2*beta*x

    g[0] += 2*gamma*(x[0] - x[1])
    g[-1] += 2*gamma*(x[-1] - x[-2])
    g[1:-1] += 2*gamma*(2*x[1:-1] - x[:-2] - x[2:])

    return g


# BFGS no line search
x = np.zeros(n)
H = np.eye(n)
res_no_ls = []
i = 0

while True:
    g = grad(x)
    p = -H @ g
    xnew = x + p

    if np.any(np.abs(xnew) > 1e6):
        print("BFGS without line search diverged")
        break

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

    if not np.isfinite(residual):
        print("BFGS without line search diverged")
        break

    if residual < eps:
        break

    if i > 1000:
        print("BFGS without line search did not converge")
        break

print("BFGS without line search")
print("Iterations =", i)
print("Status = Diverged")

# BFGS with line search
x = np.zeros(n)
H = np.eye(n)
res_ls = []
i = 0

while True:
    g = grad(x)
    p = -H @ g

    alpha_ls = 1.0
    c = 1e-4
    beta_ls = 0.5

    while f(x + alpha_ls*p) > f(x) + c*alpha_ls*(g @ p):
        alpha_ls *= beta_ls

    xnew = x + alpha_ls*p

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

# Convergence comparison
plt.figure()

if len(res_no_ls) > 0:
    plt.semilogy(res_no_ls, label="BFGS without line search")

plt.semilogy(res_ls, label="BFGS with line search")
plt.semilogy(res_scipy, label="SciPy BFGS")

plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.title("BFGS Convergence")
plt.legend()
plt.grid()
plt.show()

# Optimal profile
plt.figure()
plt.plot(np.arange(1, n+1), result.x, "o-")
plt.xlabel("Segment")
plt.ylabel("Thickness deviation")
plt.title("Optimal Fin Profile")
plt.grid()
plt.show()