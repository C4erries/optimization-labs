from utils import make_dict_cached_function
from utils.table import format_table


def build_fibonacci_until(limit):
    fib = [0, 1]
    while fib[-1] < limit:
        fib.append(fib[-1] + fib[-2])
    return fib


def fibonacci_search(func, a, b, length_limit):
    if b <= a:
        raise ValueError("Right border must be greater than left border.")
    if length_limit <= 0:
        raise ValueError("length_limit must be positive.")

    eval_f, cache, stats = make_dict_cached_function(func)
    interval_len = b - a

    if interval_len <= length_limit:
        return {
            "x_star": (a + b) / 2,
            "interval": (a, b),
            "iterations": 0,
            "history": [],
            "cache": cache,
            "stats": stats,
            "fibonacci_index": 1,
            "fibonacci_number": 1,
        }

    fib = build_fibonacci_until(interval_len / length_limit)
    n = len(fib) - 1

    x2 = a + fib[n - 1] / fib[n] * (b - a)
    x1 = a + b - x2
    fx1 = eval_f(x1)
    fx2 = eval_f(x2)

    history = []

    for k in range(n - 2):
        if fx1 <= fx2:
            a_next = a
            b_next = x2
            decision = "f(x1) <= f(x2) -> [a, x2]"
        else:
            a_next = x1
            b_next = b
            decision = "f(x1) > f(x2) -> [x1, b]"

        history.append(
            {
                "k": k,
                "a": a,
                "b": b,
                "x1": x1,
                "x2": x2,
                "fx1": fx1,
                "fx2": fx2,
                "decision": decision,
                "next_interval": f"[{a_next:.6f}, {b_next:.6f}]",
                "new_length": b_next - a_next,
            }
        )

        if k == n - 3:
            return {
                "x_star": (a_next + b_next) / 2,
                "interval": (a_next, b_next),
                "iterations": k + 1,
                "history": history,
                "cache": cache,
                "stats": stats,
                "fibonacci_index": n,
                "fibonacci_number": fib[n],
            }

        if fx1 <= fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + b - x2
            fx1 = eval_f(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + b - x1
            fx2 = eval_f(x2)


eps = 0.5
a0 = 0
b0 = 10
l = eps * 2


def f(x):
    return 2 * x * x - 12 * x + 19


result = fibonacci_search(f, a0, b0, l)

print(
    f"Chosen Fibonacci number: F_{result['fibonacci_index']} =",
    result["fibonacci_number"],
)
print(f"Iterations: {result['iterations']}")
print(
    f"Approximate solution x* ~= {result['x_star']}",
    f"+-{(result['interval'][1] - result['interval'][0]) / 2}",
)
print(
    "Uncertainty interval:",
    f"[{result['interval'][0]}, {result['interval'][1]}]",
)
print(
    "Function evaluations:",
    f"total requests = {result['stats']['requests']},",
    f"computed(N) = {result['stats']['computed']},",
)
print("Steps:")
print(
    format_table(
        result["history"],
        [
            {"key": "k", "title": "k", "align": "right"},
            {"key": "a", "title": "a", "align": "right"},
            {"key": "b", "title": "b", "align": "right"},
            {"key": "x1", "title": "x1", "align": "right"},
            {"key": "x2", "title": "x2", "align": "right"},
            {"key": "fx1", "title": "f(x1)", "align": "right"},
            {"key": "fx2", "title": "f(x2)", "align": "right"},
            {"key": "decision", "title": "decision"},
            {"key": "next_interval", "title": "next [a, b]"},
            {"key": "new_length", "title": "L", "align": "right"},
        ],
    )
)
