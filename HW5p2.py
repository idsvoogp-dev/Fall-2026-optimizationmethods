
import numpy as np
from scipy.optimize import minimize

def cost(x):
    Ts, Th, R, L = x
    return (0.6224*Ts*R*L
            + 1.7781*Th*R**2
            + 3.1661*Ts**2*L
            + 19.84*Ts**2*R)

def h(x):
    Ts, Th, R, L = x
    return (np.pi*R**2*L + (4/3)*np.pi*R**3
            - 1296000)/1296000

def g(x):
    Ts, Th, R, L = x
    return np.array([
        (0.0193*R - Ts)/10,
        (0.00954*R - Th)/10,
        (L - 240)/240,
        (10 - L)/240,
        0.0625 - Ts,
        0.0625 - Th,
        (10 - R)/10
    ])

# penalty
def penalty(x, r):
    violations = np.maximum(g(x), 0)
    return cost(x)/1000 + 0.5*r*(
        h(x)**2 + np.sum(violations**2)
    )

# Initial guess
x = np.array([1.0, 0.5, 40.0, 150.0])

# Bounds
bounds = [
    (0.0625, None),
    (0.0625, None),
    (10, None),
    (10, 240)
]


r_values = [1, 10, 100, 1000, 10000,
            100000, 1000000, 10000000]

print("QUADRATIC PENALTY METHOD")
print(f"{'r':>10} {'Ts':>10} {'Th':>10} "
      f"{'R':>10} {'L':>10} {'Cost':>12}")

for r in r_values:
    result = minimize(
        penalty, x, args=(r,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 20000, 'ftol': 1e-14}
    )

    x = result.x
    Ts, Th, R, L = x

    print(f"{r:10.0f} {Ts:10.5f} {Th:10.5f} "
          f"{R:10.5f} {L:10.5f} {cost(x):12.4f}")

print("\nFinal Quadratic Penalty Results")
print(f"Ts = {Ts:.6f} in")
print(f"Th = {Th:.6f} in")
print(f"R  = {R:.6f} in")
print(f"L  = {L:.6f} in")
print(f"Minimum cost = {cost(x):.6f}")
print(f"Volume constraint error = {h(x)*1296000:.6f} in^3")



#BARRIER
def cost(x):
    Ts, Th, R, L = x
    return (0.6224*Ts*R*L
            + 1.7781*Th*R**2
            + 3.1661*Ts**2*L
            + 19.84*Ts**2*R)


def h(x):
    Ts, Th, R, L = x
    return (np.pi*R**2*L + (4/3)*np.pi*R**3
            - 1296000)/1296000


def slack(x):
    Ts, Th, R, L = x
    return np.array([
        Ts - 0.0193*R,
        Th - 0.00954*R,
        240 - L,
        L - 10,
        Ts - 0.0625,
        Th - 0.0625,
        R - 10
    ])


def barrier(x, mu):
    s = slack(x)


    if np.any(s <= 0):
        return 1e100

    scale = np.array([10, 10, 240, 240, 1, 1, 10])

    return cost(x)/1000 - mu*np.sum(np.log(s/scale))


R = 40
L = (1296000 - (4/3)*np.pi*R**3)/(np.pi*R**2)

x = np.array([1.0, 0.5, R, L])


bounds = [
    (0.062500001, None),
    (0.062500001, None),
    (10.000001, None),
    (10.000001, 239.999999)
]

constraints = [
    {'type': 'eq', 'fun': h}
]


mu_values = [1, 0.1, 0.01, 0.001,
             0.0001, 0.00001, 0.000001]

print("BARRIER METHOD")
print(f"{'mu':>10} {'Ts':>10} {'Th':>10} "
      f"{'R':>10} {'L':>10} {'Cost':>12}")

for mu in mu_values:
    result = minimize(
        barrier, x, args=(mu,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-12}
    )

    if not result.success:
        print("Optimization warning:", result.message)

    x = result.x
    Ts, Th, R, L = x

    print(f"{mu:10.6f} {Ts:10.5f} {Th:10.5f} "
          f"{R:10.5f} {L:10.5f} {cost(x):12.4f}")

print("\nFinal Barrier Method Results")
print(f"Ts = {Ts:.6f} in")
print(f"Th = {Th:.6f} in")
print(f"R  = {R:.6f} in")
print(f"L  = {L:.6f} in")
print(f"Minimum cost = {cost(x):.6f}")
print(f"Volume constraint error = {h(x)*1296000:.6f} in^3")
