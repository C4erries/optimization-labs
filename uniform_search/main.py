import numpy as np


eps = 0.5
a0 = 0
b0 = 10
l = eps * 2


def f(x):
    return 2 * x * x - 12 * x


def uniform_search(func, a, b, n):
    step = (b - a) / (n + 1)
    x_points = np.array([a + i * step for i in range(1, n + 1)])
    f_values = np.array([func(xi) for xi in x_points])

    k = int(np.argmin(f_values))
    xk = x_points[k]

    left = a if k == 0 else x_points[k - 1]
    right = b if k == n - 1 else x_points[k + 1]

    return {
        "x_points": x_points,
        "f_values": f_values,
        "xk": xk,
        "interval": (left, right),
        "step": step,
    }


# Choose N so the uncertainty interval length does not exceed l.
n = int(np.ceil(2 * (b0 - a0) / l - 1))
result = uniform_search(f, a0, b0, n)

print(f"N = {n}")
print("Points x_i:", result["x_points"])
print("Values f(x_i):", result["f_values"])
print(f"Approximate solution x* ~= {result['xk']}")
print(
    "Uncertainty interval:",
    f"[{result['interval'][0]}, {result['interval'][1]}]",
)
