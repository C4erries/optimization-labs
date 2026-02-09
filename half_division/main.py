eps = 0.5
a0 = 0
b0 = 10
l = eps * 2


def f(x):
    return 2 * x * x - 12 * x


def half_division_search(func, a, b, length_limit):
    if b <= a:
        raise ValueError("Right border must be greater than left border.")
    if length_limit <= 0:
        raise ValueError("length_limit must be positive.")

    k = 0
    xc = (a + b) / 2
    history = []

    while True:
        interval_len = b - a
        y = a + interval_len / 4
        z = b - interval_len / 4

        fy = func(y)
        fxc = func(xc)
        fz = func(z)

        if fy < fxc:
            a_next = a
            b_next = xc
            xc_next = y
            decision = "fy < fxc -> [a, xc]"
        elif fz < fxc:
            a_next = xc
            b_next = b
            xc_next = z
            decision = "fz < fxc -> [xc, b]"
        else:
            a_next = y
            b_next = z
            xc_next = xc
            decision = "fxc is min -> [y, z]"

        new_len = b_next - a_next
        history.append(
            {
                "k": k,
                "a": a,
                "b": b,
                "xc": xc,
                "y": y,
                "z": z,
                "fy": fy,
                "fxc": fxc,
                "fz": fz,
                "decision": decision,
                "new_interval": (a_next, b_next),
                "new_length": new_len,
            }
        )

        if new_len <= length_limit:
            return {
                "x_star": xc_next,
                "interval": (a_next, b_next),
                "iterations": k + 1,
                "history": history,
            }

        a = a_next
        b = b_next
        xc = xc_next
        k += 1


result = half_division_search(f, a0, b0, l)

print(f"Iterations: {result['iterations']}")
print(f"Approximate solution x* ~= {result['x_star']}")
print(
    "Uncertainty interval:",
    f"[{result['interval'][0]}, {result['interval'][1]}]",
)
