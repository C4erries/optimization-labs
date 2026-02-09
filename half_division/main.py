from utils import make_dict_cached_function



def f(x):
    return 2 * x * x - 12 * x


def half_division_search(func, a, b, length_limit):
    if b <= a:
        raise ValueError("Right border must be greater than left border.")
    if length_limit <= 0:
        raise ValueError("length_limit must be positive.")

    eval_f, cache, stats = make_dict_cached_function(func)

    k = 0
    x = (a + b) / 2
    history = []

    while True:
        interval_len = b - a
        y = a + interval_len / 4
        z = b - interval_len / 4

        fy = eval_f(y)
        fx = eval_f(x)
        fz = eval_f(z)

        if fy < fx:
            a_next = a
            b_next = x
            x_next = y
            decision = "fy < fx -> [a, x]"
        elif fz < fx:
            a_next = x
            b_next = b
            x_next = z
            decision = "fz < fx -> [x, b]"
        else:
            a_next = y
            b_next = z
            x_next = x
            decision = "fx is min -> [y, z]"

        new_len = b_next - a_next
        history.append(
            {
                "k": k,
                "a": a,
                "b": b,
                "x": x,
                "y": y,
                "z": z,
                "fy": fy,
                "fx": fx,
                "fz": fz,
                "decision": decision,
                "new_interval": (a_next, b_next),
                "new_length": new_len,
            }
        )

        if new_len <= length_limit:
            return {
                "x_star": x_next,
                "interval": (a_next, b_next),
                "iterations": k + 1,
                "history": history,
                "cache": cache,
                "stats": stats,
            }

        a = a_next
        b = b_next
        x = x_next
        k += 1

eps = 0.5
a0 = 0
b0 = 10
l = eps * 2


result = half_division_search(f, a0, b0, l)

print(f"Iterations: {result['iterations']}")
print(f"Approximate solution x* ~= {result['x_star']}")
print(
    "Uncertainty interval:",
    f"[{result['interval'][0]}, {result['interval'][1]}]",
)
print(
    "Function evaluations:",
    f"total requests = {result['stats']['requests']},",
    f"computed = {result['stats']['computed']},",
    f"saved = {result['stats']['requests'] - result['stats']['computed']}",
)
print(f"DP cache size: {len(result['cache'])}")
