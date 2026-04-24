import numpy as np

from davidson_fletcher_powell.main import davidson_fletcher_powell
from utils import euclidean_norm, format_vector, make_cached_nd_function
from utils.table import format_table


def _as_constraint_list(constraints):
    if callable(constraints):
        return [constraints]
    return list(constraints)


def penalty_method(
    func,
    constraints,
    x0,
    eps1,
    eps2,
    delta,
    M,
    r0=1.0,
    c=10.0,
    inner_M=100,
):
    if eps1 <= 0:
        raise ValueError("eps1 must be positive.")
    if eps2 <= 0:
        raise ValueError("eps2 must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")
    if r0 <= 0:
        raise ValueError("r0 must be positive.")
    if c <= 1:
        raise ValueError("c must be greater than 1.")

    constraint_functions = _as_constraint_list(constraints)
    eval_f, cache_f, stats_f = make_cached_nd_function(func)
    cached_constraints = [make_cached_nd_function(g) for g in constraint_functions]
    eval_constraints = [item[0] for item in cached_constraints]

    x = np.asarray(x0, dtype=float).reshape(-1)
    r = float(r0)
    history = []
    k = 0
    prev_eps2_satisfied = False

    while True:
        fx = eval_f(x)
        gx = np.array([eval_g(x) for eval_g in eval_constraints], dtype=float)
        constraint_norm = euclidean_norm(gx)
        penalty_value = 0.5 * r * constraint_norm * constraint_norm

        if constraint_norm <= eps1 and prev_eps2_satisfied:
            history.append(
                {
                    "k": k,
                    "r_k": r,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "g_norm": constraint_norm,
                    "penalty": penalty_value,
                    "inner_iter": "-",
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "df_abs": 0.0,
                    "decision": "constraints and eps2",
                }
            )
            return _make_result(
                x, fx, gx, history, cache_f, stats_f, cached_constraints, "constraints"
            )

        if k >= M:
            history.append(
                {
                    "k": k,
                    "r_k": r,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "g_norm": constraint_norm,
                    "penalty": penalty_value,
                    "inner_iter": "-",
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "df_abs": 0.0,
                    "decision": "k >= M",
                }
            )
            return _make_result(
                x, fx, gx, history, cache_f, stats_f, cached_constraints, "max_iterations"
            )

        def penalty_function(point):
            values = np.array(
                [eval_g(point) for eval_g in eval_constraints],
                dtype=float,
            )
            return eval_f(point) + 0.5 * r * float(np.dot(values, values))

        inner = davidson_fletcher_powell(
            penalty_function,
            x,
            eps1,
            eps2,
            delta,
            inner_M,
        )

        x_next = inner["x_star"]
        f_next = eval_f(x_next)
        dx_norm = euclidean_norm(x_next - x)
        df_abs = abs(f_next - fx)
        g_next = np.array([eval_g(x_next) for eval_g in eval_constraints], dtype=float)
        g_next_norm = euclidean_norm(g_next)

        decision = "increase penalty"
        eps2_satisfied = dx_norm <= eps2 and df_abs <= eps2
        if g_next_norm <= eps1 and eps2_satisfied and prev_eps2_satisfied:
            decision = "stop"
        elif eps2_satisfied:
            decision = "eps2 x1, increase penalty"

        history.append(
            {
                "k": k,
                "r_k": r,
                "x_k": format_vector(x),
                "f_k": fx,
                "g_norm": constraint_norm,
                "penalty": penalty_value,
                "inner_iter": inner["iterations"],
                "x_next": format_vector(x_next),
                "f_next": f"{f_next:.6f}",
                "dx_norm": dx_norm,
                "df_abs": df_abs,
                "decision": decision,
            }
        )

        if g_next_norm <= eps1 and eps2_satisfied and prev_eps2_satisfied:
            return _make_result(
                x_next,
                f_next,
                g_next,
                history,
                cache_f,
                stats_f,
                cached_constraints,
                "constraints_and_point",
            )

        prev_eps2_satisfied = eps2_satisfied
        x = x_next
        r *= c
        k += 1


def _make_result(x, fx, gx, history, cache_f, stats_f, cached_constraints, reason):
    return {
        "x_star": x,
        "f_star": fx,
        "g_star": gx,
        "iterations": len(history),
        "history": history,
        "cache": {
            "f": cache_f,
            "g": [item[1] for item in cached_constraints],
        },
        "stats": {
            "f_requests": stats_f["requests"],
            "f_computed": stats_f["computed"],
            "g_requests": sum(item[2]["requests"] for item in cached_constraints),
            "g_computed": sum(item[2]["computed"] for item in cached_constraints),
        },
        "reason": reason,
    }


eps1 = 1e-4
eps2 = 1e-4
delta = 1e-6
M = 20


def f(x):
    return 3 * x[0] * x[0] + 4 * x[1] * x[1] - 2 * x[0] * x[1] + x[0]


def g(x):
    return x[0] + x[1] - 1


def main():
    result = penalty_method(
        f,
        g,
        x0=np.array([0.0, 1.0], dtype=float),
        eps1=eps1,
        eps2=eps2,
        delta=delta,
        M=M,
    )

    print(f"Iterations: {result['iterations']}")
    print(f"Stop reason: {result['reason']}")
    print(f"Approximate solution x* ~= {format_vector(result['x_star'])}")
    print(f"f(x*) ~= {result['f_star']}")
    print(f"g(x*) ~= {format_vector(result['g_star'])}")
    print(
        "Function evaluations:",
        f"f: requests = {result['stats']['f_requests']}, computed = {result['stats']['f_computed']};",
        f"g: requests = {result['stats']['g_requests']}, computed = {result['stats']['g_computed']}",
    )
    print("Steps:")
    print(
        format_table(
            result["history"],
            [
                {"key": "k", "title": "k", "align": "right"},
                {"key": "r_k", "title": "r_k", "align": "right"},
                {"key": "x_k", "title": "x_k"},
                {"key": "f_k", "title": "f(x_k)", "align": "right"},
                {"key": "g_norm", "title": "||g||_2", "align": "right"},
                {"key": "penalty", "title": "penalty", "align": "right"},
                {"key": "inner_iter", "title": "inner", "align": "right"},
                {"key": "x_next", "title": "x_{k+1}"},
                {"key": "f_next", "title": "f(x_{k+1})", "align": "right"},
                {"key": "dx_norm", "title": "||dx||_2", "align": "right"},
                {"key": "df_abs", "title": "|df|", "align": "right"},
                {"key": "decision", "title": "decision"},
            ],
        )
    )


if __name__ == "__main__":
    main()
