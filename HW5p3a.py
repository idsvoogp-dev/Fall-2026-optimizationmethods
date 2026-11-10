
import numpy as np
from scipy.optimize import minimize

# constants
L = 1000
td = 1
rho = 7.85e-3
E = 2.1e7
nu = 0.3
M = 3e8
V = 3e6
sigma_a = 16000
tau_a = 9000

# Variables: b, tf, h, tw, a


def cost(x):
    b, tf, h, tw, a = x
    return rho*(L*(2*b*tf + 2*h*tw)
                + (L/a)*b*h*td)

# inertia
def inertia(x):
    b, tf, h, tw, a = x
    return b*tf*h**2/2 + tw*h**3/6


def equality(x):
    b, tf, h, tw, a = x

    h1 = (inertia(x) - 1.2e6)/1.2e6
    h2 = (tf - 1.25*tw)/5

    return np.array([h1, h2])


def inequalities(x):
    b, tf, h, tw, a = x

    I = inertia(x)
    tau = V/(2*h*tw)

    tau_cr = (np.pi**2*E/(12*(1-nu**2))
              * (tw/h)**2
              * (5.34 + 4*(h/a)**2))

    return np.array([
        (M*h/(2*I) - sigma_a)/sigma_a,
        (tau - tau_a)/tau_a,
        (1.5*tau - tau_cr)/9000,
        (b/tf - 60)/60,
        (h/tw - 150)/150,
        (a - 2*h)/200
    ])

# Quadratic penalty function
def penalty(x, r):
    eq = equality(x)
    g = np.maximum(inequalities(x), 0)

    return cost(x)/1000 + 0.5*r*(
        np.sum(eq**2) + np.sum(g**2)
    )


bounds = [
    (20, 80),       # b
    (0.8, 6),       # tf
    (60, 200),      # h
    (0.6, 5),       # tw
    (50, 400)       # a
]

# Initial guess
x = np.array([40, 2.5, 120, 2.0, 150])


r_values = [1, 10, 100, 1000, 10000,
            100000, 1000000, 10000000]

print("QUADRATIC PENALTY METHOD")
print(f"{'r':>10} {'b':>10} {'tf':>10} "
      f"{'h':>10} {'tw':>10} {'a':>10} {'Mass':>12}")

for r in r_values:

    result = minimize(
        penalty,
        x,
        args=(r,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 50000, 'ftol': 1e-15}
    )

    x = result.x

    b, tf, h, tw, a = x

    print(f"{r:10.0f} {b:10.4f} {tf:10.4f} "
          f"{h:10.4f} {tw:10.4f} {a:10.4f} "
          f"{cost(x):12.4f}")

print("\nFINAL QUADRATIC PENALTY RESULTS")
print(f"b  = {b:.6f} cm")
print(f"tf = {tf:.6f} cm")
print(f"h  = {h:.6f} cm")
print(f"tw = {tw:.6f} cm")
print(f"a  = {a:.6f} cm")
print(f"Minimum mass = {cost(x):.6f} kg")

print("Equality constraint errors:", equality(x))
print("Inequality constraints:", inequalities(x))








