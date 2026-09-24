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

# BFGS
z = np.array([4.0, np.deg2rad(45)])
H = np.eye(2)
res1 = []
i = 0

while True:
    g = grad(z)
    p = -H @ g
    alpha = 1.0

    while True:
        znew = z + alpha*p
        znew[1] = min(max(znew[1], 1e-6), np.pi/2)

        if P(znew) < P(z):
            break
        alpha *= 0.5

    s = znew - z
    y = grad(znew) - g


    if abs(y @ s) > 1e-12:
        rho = 1/(y @ s)
        I = np.eye(2)
        H = (I-rho*np.outer(s,y)) @ H @ \
            (I-rho*np.outer(y,s)) + rho*np.outer(s,s)

    z = znew
    residual = np.linalg.norm(grad(z))
    res1.append(residual)
    i += 1

    if residual < eps:
        break

d, theta = z
theta_deg = np.rad2deg(theta)
w = A/d + 2*d/np.tan(theta)

print("My BFGS")
print("Iterations =", i)
print("d =", d, "m")
print("theta =", theta_deg, "degrees")
print("w =", w, "m")
print("P =", P(z), "m")
print("Residual =", residual)

# SciPy BFGS
res2 = []

def callback(z):
    res2.append(np.linalg.norm(grad(z)))

result = minimize(
    P, [4, np.deg2rad(45)],
    jac=grad,
    method="L-BFGS-B",
    bounds=[(1e-6, None), (1e-6, np.pi/2)],
    callback=callback,
    options={"gtol": eps}
)

d2, theta2 = result.x
w2 = A/d2 + 2*d2/np.tan(theta2)

print("\nSciPy BFGS")
print("Iterations =", result.nit)
print("d =", d2, "m")
print("theta =", np.rad2deg(theta2), "degrees")
print("w =", w2, "m")
print("P =", result.fun)
print("Residual =", np.linalg.norm(grad(result.x)))


dvals = np.linspace(3, 7, 100)
tvals = np.linspace(np.deg2rad(30), np.deg2rad(90), 100)
D, T = np.meshgrid(dvals, tvals)
Z = A/D + 2*D/np.sin(T)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(D, np.rad2deg(T), Z)
ax.scatter(d, theta_deg, P(z), color="red")
ax.set_xlabel("d (m)")
ax.set_ylabel("theta (degrees)")
ax.set_zlabel("P (m)")
plt.show()

# Contour plot
plt.contour(D, np.rad2deg(T), Z, levels=20)
plt.plot(d, theta_deg, "ro")
plt.xlabel("d (m)")
plt.ylabel("theta (degrees)")
plt.title("Wetted Perimeter")
plt.grid()
plt.show()

