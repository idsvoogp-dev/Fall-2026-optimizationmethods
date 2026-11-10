
import numpy as np
#lagrange
A = np.array([
    [0.10, 0.00, 1.0],
    [0.00, 0.20, 1.0],
    [1.00, 1.00, 0.0]
])

b = np.array([-10.0, -15.0, 100.0])

P1, P2, lam = np.linalg.solve(A, b)

# total fuel cost
f1 = 0.05*P1**2 + 10*P1
f2 = 0.10*P2**2 + 15*P2
cost = f1 + f2

print("Lagrange Multiplier Method")
print(f"P1 = {P1:.4f} MW")
print(f"P2 = {P2:.4f} MW")
print(f"Lambda = {lam:.4f}")
print(f"Total cost = {cost:.4f} per hour")
print(f"Power balance = {P1 + P2:.4f} MW")




# Penalty
r_values = [1, 10, 100, 1000, 100000]

print("Quadratic Penalty Method")
print(f"{'r':>10} {'P1 (MW)':>12} {'P2 (MW)':>12} "
      f"{'Total Cost':>14} {'Total Power':>14}")

for r in r_values:

    A = np.array([
        [0.10 + r, r],
        [r, 0.20 + r]
    ])

    b = np.array([100*r - 10, 100*r - 15])

    P1, P2 = np.linalg.solve(A, b)

    # total fuel cost
    f1 = 0.05*P1**2 + 10*P1
    f2 = 0.10*P2**2 + 15*P2
    cost = f1 + f2

    print(f"{r:10.0f} {P1:12.4f} {P2:12.4f} "
          f"{cost:14.4f} {P1 + P2:14.4f}")

