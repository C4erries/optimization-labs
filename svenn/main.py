import numpy as np

from utils import make_dict_cached_function
from utils.table import format_table


def svenn_search(func, x0, step, max_iterations=50):
    if step <= 0:
        raise ValueError("step must be positive.")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    eval_f, cache, stats = make_dict_cached_function(func)
    history = []

    x_left = x0 - step
    x_mid = x0
    x_right = x0 + step

    f_left = eval_f(x_left)
    f_mid = eval_f(x_mid)
    f_right = eval_f(x_right)

    if f_left >= f_mid <= f_right:
        interval = (x_left, x_right)
        history.append(
            {
                "k": 0,
                "x_prev": x_left,
                "x_curr": x_mid,
                "x_next": x_right,
                "f_prev": f_left,
                "f_curr": f_mid,
                "f_next": f_right,
                "step": step,
                "decision": "f(x0-step) >= f(x0) <= f(x0+step)",
                "interval": f"[{interval[0]:.6f}, {interval[1]:.6f}]",
            }
        )
        return {
            "interval": interval,
            "midpoint": (interval[0] + interval[1]) / 2,
            "iterations": 1,
            "history": history,
            "cache": cache,
            "stats": stats,
        }

    if f_left <= f_mid >= f_right:
        raise ValueError(
            "Initial step is too large or the function is not unimodal near x0."
        )

    direction = 1 if f_right < f_mid else -1
    direction_label = "right" if direction > 0 else "left"
    history.append(
        {
            "k": 0,
            "x_prev": x_left,
            "x_curr": x_mid,
            "x_next": x_right,
            "f_prev": f_left,
            "f_curr": f_mid,
            "f_next": f_right,
            "step": step,
            "decision": f"choose {direction_label} direction",
            "interval": "-",
        }
    )

    x_prev = x_mid
    f_prev = f_mid
    x_curr = x0 + direction * step
    f_curr = f_right if direction > 0 else f_left

    current_step = step

    for k in range(1, max_iterations + 1):
        current_step *= 2
        x_next = x0 + direction * current_step
        f_next = eval_f(x_next)

        if f_next >= f_curr:
            interval = (min(x_prev, x_next), max(x_prev, x_next))
            decision = (
                f"f(x_next) >= f(x_curr) -> [{interval[0]:.6f}, {interval[1]:.6f}]"
            )
        else:
            interval = None
            decision = f"f(x_next) < f(x_curr) -> continue {direction_label}"

        history.append(
            {
                "k": k,
                "x_prev": x_prev,
                "x_curr": x_curr,
                "x_next": x_next,
                "f_prev": f_prev,
                "f_curr": f_curr,
                "f_next": f_next,
                "step": abs(x_next - x_curr),
                "decision": decision,
                "interval": (
                    f"[{interval[0]:.6f}, {interval[1]:.6f}]"
                    if interval is not None
                    else "-"
                ),
            }
        )

        if interval is not None:
            return {
                "interval": interval,
                "midpoint": (interval[0] + interval[1]) / 2,
                "iterations": k + 1,
                "history": history,
                "cache": cache,
                "stats": stats,
            }

        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next
        f_curr = f_next

    raise RuntimeError("Failed to bracket a minimum within max_iterations.")


x0 = 100.0
step0 = 0.1


def f(x):
    return 2 * x * x - 12 * x + 19


result = svenn_search(f, x0, step0)

print(f"Iterations: {result['iterations']}")
print(
    "Bracketing interval:",
    f"[{result['interval'][0]}, {result['interval'][1]}]",
)
print(f"Midpoint x ~= {result['midpoint']}")
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
            {"key": "x_prev", "title": "x_prev", "align": "right"},
            {"key": "x_curr", "title": "x_curr", "align": "right"},
            {"key": "x_next", "title": "x_next", "align": "right"},
            {"key": "f_prev", "title": "f(x_prev)", "align": "right"},
            {"key": "f_curr", "title": "f(x_curr)", "align": "right"},
            {"key": "f_next", "title": "f(x_next)", "align": "right"},
            {"key": "step", "title": "step", "align": "right"},
            {"key": "decision", "title": "decision"},
            {"key": "interval", "title": "interval"},
        ],
    )
)
