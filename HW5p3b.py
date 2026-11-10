import numpy as np
from scipy.optimize import minimize


L = 1000
td = 1
rho = 7.85e-3
E = 2.1e7
nu = 0.3
M = 3e8
V = 3e6
sigma_a = 16000
tau_a = 9000

def cost(x):
    b, tf, h, tw, a = x
    return rho*(L*(2*b*tf + 2*h*tw)
                + (L/a)*b*h*td)

# Moment of inertia
def inertia(x):
    b, tf, h, tw, a = x
    return b*tf*h**2/2 + tw*h**3/6

# Equality constraints
def equality(x):
    b, tf, h, tw, a = x

    h1 = (inertia(x) - 1.2e6)/1.2e6
    h2 = (tf - 1.25*tw)/5

    return np.array([h1, h2])

# Inequality constraints: g(x) <= 0
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

# Logarithmic barrier function
def barrier(x, mu):

    g = inequalities(x)

    # Must stay strictly inside all inequalities
    if np.any(g >= 0):
        return 1e100

    return cost(x)/1000 - mu*np.sum(np.log(-g))

# Variable bounds
bounds = [
    (20, 80),
    (0.8, 6),
    (60, 200),
    (0.6, 5),
    (50, 400)
]

# Given initial guess
x0 = np.array([40, 2.5, 120, 2.0, 150])

# Step 1: Find a strictly feasible starting point
def distance(x):
    scale = np.array([40, 2.5, 120, 2, 150])
    return np.sum(((x-x0)/scale)**2)

feasible = minimize(
    distance,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=[
        {'type': 'eq', 'fun': equality},
        {'type': 'ineq',
         'fun': lambda x: -inequalities(x) - 1e-5}
    ],
    options={'maxiter': 10000, 'ftol': 1e-12}
)

if not feasible.success:
    raise RuntimeError("Could not find feasible starting point")

x = feasible.x

print("Feasible starting point:")
print(x)

# Step 2: Decrease barrier parameter
mu_values = [1, 0.1, 0.01, 0.001,
             0.0001, 0.00001, 0.000001]

print("\nBARRIER METHOD")
print(f"{'mu':>10} {'b':>10} {'tf':>10} "
      f"{'h':>10} {'tw':>10} {'a':>10} {'Mass':>12}")

for mu in mu_values:

    result = minimize(
        barrier,
        x,
        args=(mu,),
        method='SLSQP',
        bounds=bounds,
        constraints=[
            {'type': 'eq', 'fun': equality},
            {'type': 'ineq',
             'fun': lambda x: -inequalities(x)}
        ],
        options={'maxiter': 10000, 'ftol': 1e-12}
    )

    if not result.success:
        print("Warning:", result.message)

    x = result.x

    b, tf, h, tw, a = x

    print(f"{mu:10.6f} {b:10.4f} {tf:10.4f} "
          f"{h:10.4f} {tw:10.4f} {a:10.4f} "
          f"{cost(x):12.4f}")

print("\nFINAL BARRIER RESULTS")
print(f"b  = {b:.6f} cm")
print(f"tf = {tf:.6f} cm")
print(f"h  = {h:.6f} cm")
print(f"tw = {tw:.6f} cm")
print(f"a  = {a:.6f} cm")
print(f"Minimum mass = {cost(x):.6f} kg")

print("Equality constraint errors:", equality(x))
print("Inequality constraints:", inequalities(x))