import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

A = 50
eps = 1e-6

def P(z):
    d, theta = z
    return A/d + 2*d/np.sin(theta)

def grad(z):
    d, theta = z
    return np.array([
        -A/d**2 + 2/np.sin(theta),
        -2*d*np.cos(theta)/np.sin(theta)**2
    ])

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

A = 50
eps = 1e-6

def P(z):
    d, theta = z
    return A/d - d/np.tan(theta) + 2*d/np.sin(theta)

def grad(z):
    d, theta = z
    return np.array([
        -A/d**2 - 1/np.tan(theta) + 2/np.sin(theta),
        d/np.sin(theta)**2 - 2*d*np.cos(theta)/np.sin(theta)**2
    ])


# BFGS WITHOUT line search


z = np.array([4.0, np.deg2rad(45)])
H = np.eye(2)
res_no_ls = []
i = 0

while True:
    g = grad(z)
    p = -H @ g
    alpha = 1.0

    znew = z + alpha*p
    znew[0] = max(znew[0], 1e-6)
    znew[1] = np.clip(znew[1], 1e-6, np.pi/2)

    s = znew - z
    y = grad(znew) - g

    if abs(y @ s) > 1e-12:
        rho = 1/(y @ s)
        I = np.eye(2)
        H = (I-rho*np.outer(s,y)) @ H @ \
            (I-rho*np.outer(y,s)) + rho*np.outer(s,s)

    z = znew
    residual = np.linalg.norm(grad(z))
    res_no_ls.append(residual)
    i += 1

    if residual < eps:
        break

d_no, theta_no = z

print("BFGS WITHOUT Line Search")
print("Iterations =", i)
print("d =", d_no)
print("theta =", np.rad2deg(theta_no))
print("P =", P(z))
print("Residual =", residual)



# BFGS WITH line search


z = np.array([4.0, np.deg2rad(45)])
H = np.eye(2)
res_ls = []
i = 0

while True:
    g = grad(z)
    p = -H @ g

    alpha = 1.0
    c = 1e-4
    beta = 0.5

    while True:
        znew = z + alpha*p
        znew[0] = max(znew[0], 1e-6)
        znew[1] = np.clip(znew[1], 1e-6, np.pi/2)

        if P(znew) <= P(z) + c*alpha*(g @ p):
            break

        alpha *= beta

    s = znew - z
    y = grad(znew) - g

    if abs(y @ s) > 1e-12:
        rho = 1/(y @ s)
        I = np.eye(2)
        H = (I-rho*np.outer(s,y)) @ H @ \
            (I-rho*np.outer(y,s)) + rho*np.outer(s,s)

    z = znew
    residual = np.linalg.norm(grad(z))
    res_ls.append(residual)
    i += 1

    if residual < eps:
        break

d_ls, theta_ls = z

print("\nBFGS WITH Line Search")
print("Iterations =", i)
print("d =", d_ls)
print("theta =", np.rad2deg(theta_ls))
print("P =", P(z))
print("Residual =", residual)


# 3. SciPy BFGS

res_scipy = []

def callback(z):
    res_scipy.append(np.linalg.norm(grad(z)))

result = minimize(
    P,
    [4, np.deg2rad(45)],
    jac=grad,
    method="BFGS",
    callback=callback,
    options={"gtol": eps}
)

d_sp, theta_sp = result.x

print("\nSciPy BFGS")
print("Iterations =", result.nit)
print("d =", d_sp)
print("theta =", np.rad2deg(theta_sp))
print("P =", result.fun)
print("Residual =", np.linalg.norm(grad(result.x)))


# comparison


plt.figure()
plt.semilogy(res_no_ls, label="BFGS without line search")
plt.semilogy(res_ls, label="BFGS with line search")
plt.semilogy(res_scipy, label="SciPy BFGS")
plt.xlabel("Iteration")
plt.ylabel("Residual")
plt.title("BFGS Convergence Comparison")
plt.legend()
plt.grid()
plt.show()