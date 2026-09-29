import math
eps = 1e-6


h = 4
d = 4

def L(x):
    return math.sqrt((x+d)**2 + (h*(x+d)/x)**2)

# Golden section
xl = 0.01
xu = 20
R = (math.sqrt(5)-1)/2
i = 0

while True:
    x1 = xu - R*(xu-xl)
    x2 = xl + R*(xu-xl)

    if L(x1) < L(x2):
        xu = x2
    else:
        xl = x1

    xopt = (xl+xu)/2
    residual = (xu-xl)/max(abs(xopt),1)
    i += 1

    if residual < eps:
        break

print("Golden Section Search")
print("Initial guesses: xl =", 0.01, ", xu =", 20)
print("Iterations =", i)
print("Distance from fence =", xopt, "m")
print("Minimum ladder length =", L(xopt), "m")
print("Residual =", residual)