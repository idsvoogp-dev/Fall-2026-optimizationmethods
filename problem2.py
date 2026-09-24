import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

E = 2e11
A = 0.0001
w = 0.44
l = 0.56
h = 0.5
F = 10000
eps = 1e-6

def solve(theta):
    t = np.deg2rad(theta)
    ax = E*A/l*(w/(2*l))**2
    ay = E*A/l*(h/l)**2

    def V(z):
        x, y = z
        return ax*x**2 + ay*y**2 - F*x*np.cos(t) - F*y*np.sin(t)

    def grad(z):
        x, y = z
        return np.array([2*ax*x - F*np.cos(t),
                         2*ay*y - F*np.sin(t)])

    # BFGS
    z = np.array([0., 0.])
    H = np.eye(2)
    hist1 = [np.linalg.norm(grad(z))]
    i = 0

    while hist1[-1] > eps:
        g = grad(z)
        d = -H @ g
        alpha = -(g @ d)/(2*ax*d[0]**2 + 2*ay*d[1]**2)

        znew = z + alpha*d
        s = znew - z
        yk = grad(znew) - g

        rho = 1/(yk @ s)
        H = (np.eye(2) - rho*np.outer(s,yk)) @ H @ \
            (np.eye(2) - rho*np.outer(yk,s)) + rho*np.outer(s,s)

        z = znew
        hist1.append(np.linalg.norm(grad(z)))
        i += 1

    # SciPy BFGS
    hist2 = []

    def callback(z):
        hist2.append(np.linalg.norm(grad(z)))

    result = minimize(V, [0., 0.], jac=grad, method="BFGS",
                      callback=callback,
                      options={"gtol": eps})

    print("\ntheta =", theta, "degrees")
    print("In-house BFGS:", i, "iterations")
    print("x,y =", z)
    print("V =", V(z))
    print("Residual =", hist1[-1])

    print("SciPy BFGS:", result.nit, "iterations")
    print("x,y =", result.x)
    print("V =", result.fun)
    print("Residual =", np.linalg.norm(grad(result.x)))

    return V, z, result.x, hist1, hist2


results = {}

for theta in range(0, 91, 15):
    results[theta] = solve(theta)

# 45 degree plot
V, z, zscipy, hist1, hist2 = results[45]

x0, y0 = z
x = np.linspace(x0-0.05, x0+0.05, 100)
y = np.linspace(y0-0.05, y0+0.05, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[V([a,b]) for a in x] for b in y])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(X, Y, Z)
ax.scatter(x0, y0, V(z), color="red")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("V(x,y)")
plt.show()

plt.contour(X, Y, Z, levels=20)
plt.plot(x0, y0, "ro")
plt.xlabel("x")
plt.ylabel("y")
plt.title("V(x,y), theta = 45 degrees")
plt.grid()
plt.show()