import numpy as np

from utils import euclidean_norm, format_vector, make_cached_nd_function, numerical_gradient, numerical_hessian
from utils.table import format_table


def solve_lagrange_multiplier(
    func,
    constraint,
    x0,
    lambda0,
    eps1,
    eps2,
    delta,
    M,
):
    if eps1 <= 0:
        raise ValueError("eps1 must be positive.")
    if eps2 <= 0:
        raise ValueError("eps2 must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")

    eval_f, cache_f, stats_f = make_cached_nd_function(func)
    eval_g, cache_g, stats_g = make_cached_nd_function(constraint)

    x = np.asarray(x0, dtype=float).reshape(-1)
    lambda_k = float(lambda0)
    history = []
    k = 0
    prev_eps2_satisfied = False

    while True:
        fx = eval_f(x)
        gx = eval_g(x)
        grad_f = numerical_gradient(eval_f, x, delta)
        grad_g = numerical_gradient(eval_g, x, delta)
        grad_l = grad_f + lambda_k * grad_g
        residual = np.concatenate([grad_l, np.array([gx], dtype=float)])
        residual_norm = euclidean_norm(residual)

        if residual_norm <= eps1:
            history.append(
                {
                    "k": k,
                    "x_k": format_vector(x),
                    "lambda_k": lambda_k,
                    "f_k": fx,
                    "g_k": gx,
                    "res_norm": residual_norm,
                    "x_next": format_vector(x),
                    "lambda_next": lambda_k,
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "dlambda_abs": 0.0,
                    "decision": "residual <= eps1",
                }
            )
            return {
                "x_star": x,
                "lambda_star": lambda_k,
                "f_star": fx,
                "g_star": gx,
                "iterations": len(history),
                "history": history,
                "stats": {
                    "f_requests": stats_f["requests"],
                    "f_computed": stats_f["computed"],
                    "g_requests": stats_g["requests"],
                    "g_computed": stats_g["computed"],
                },
                "cache": {"f": cache_f, "g": cache_g},
                "reason": "residual",
            }

        if k >= M:
            history.append(
                {
                    "k": k,
                    "x_k": format_vector(x),
                    "lambda_k": lambda_k,
                    "f_k": fx,
                    "g_k": gx,
                    "res_norm": residual_norm,
                    "x_next": format_vector(x),
                    "lambda_next": lambda_k,
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "dlambda_abs": 0.0,
                    "decision": "k >= M",
                }
            )
            return {
                "x_star": x,
                "lambda_star": lambda_k,
                "f_star": fx,
                "g_star": gx,
                "iterations": len(history),
                "history": history,
                "stats": {
                    "f_requests": stats_f["requests"],
                    "f_computed": stats_f["computed"],
                    "g_requests": stats_g["requests"],
                    "g_computed": stats_g["computed"],
                },
                "cache": {"f": cache_f, "g": cache_g},
                "reason": "max_iterations",
            }

        hessian_f = numerical_hessian(eval_f, x, delta)
        hessian_g = numerical_hessian(eval_g, x, delta)
        hessian_l = hessian_f + lambda_k * hessian_g

        system_matrix = np.block(
            [
                [hessian_l, grad_g.reshape(-1, 1)],
                [grad_g.reshape(1, -1), np.zeros((1, 1), dtype=float)],
            ]
        )

        try:
            step = np.linalg.solve(system_matrix, -residual)
        except np.linalg.LinAlgError:
            history.append(
                {
                    "k": k,
                    "x_k": format_vector(x),
                    "lambda_k": lambda_k,
                    "f_k": fx,
                    "g_k": gx,
                    "res_norm": residual_norm,
                    "x_next": format_vector(x),
                    "lambda_next": lambda_k,
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "dlambda_abs": 0.0,
                    "decision": "singular KKT matrix",
                }
            )
            return {
                "x_star": x,
                "lambda_star": lambda_k,
                "f_star": fx,
                "g_star": gx,
                "iterations": len(history),
                "history": history,
                "stats": {
                    "f_requests": stats_f["requests"],
                    "f_computed": stats_f["computed"],
                    "g_requests": stats_g["requests"],
                    "g_computed": stats_g["computed"],
                },
                "cache": {"f": cache_f, "g": cache_g},
                "reason": "singular_kkt",
            }

        x_next = x + step[:-1]
        lambda_next = lambda_k + step[-1]
        f_next = eval_f(x_next)
        dx_norm = euclidean_norm(x_next - x)
        dlambda_abs = abs(lambda_next - lambda_k)

        decision = "continue"
        eps2_satisfied = dx_norm <= eps2 and dlambda_abs <= eps2
        if eps2_satisfied and prev_eps2_satisfied:
            decision = "eps2 x2 -> stop"
        elif eps2_satisfied:
            decision = "eps2 x1"

        history.append(
            {
                "k": k,
                "x_k": format_vector(x),
                "lambda_k": lambda_k,
                "f_k": fx,
                "g_k": gx,
                "res_norm": residual_norm,
                "x_next": format_vector(x_next),
                "lambda_next": lambda_next,
                "f_next": f"{f_next:.6f}",
                "dx_norm": dx_norm,
                "dlambda_abs": dlambda_abs,
                "decision": decision,
            }
        )

        if eps2_satisfied and prev_eps2_satisfied:
            g_next = eval_g(x_next)
            return {
                "x_star": x_next,
                "lambda_star": lambda_next,
                "f_star": f_next,
                "g_star": g_next,
                "iterations": len(history),
                "history": history,
                "stats": {
                    "f_requests": stats_f["requests"],
                    "f_computed": stats_f["computed"],
                    "g_requests": stats_g["requests"],
                    "g_computed": stats_g["computed"],
                },
                "cache": {"f": cache_f, "g": cache_g},
                "reason": "point_and_lambda_twice",
            }

        prev_eps2_satisfied = eps2_satisfied
        x = x_next
        lambda_k = lambda_next
        k += 1


eps1 = 1e-4
eps2 = 1e-4
delta = 1e-6
M = 100


def f_a(x):
    return 3 * x[0] * x[0] + 4 * x[1] * x[1] - 2 * x[0] * x[1] + x[0]


def g_a(x):
    return x[0] + x[1] - 1


def f_b(x):
    return 5 * x[0] * x[0] + x[1] * x[1] - x[0] * x[1] + x[0]


def g_b(x):
    return 3 * x[0] + 2 * x[1] - 1


def print_result(title, result):
    print(title)
    print(f"Iterations: {result['iterations']}")
    print(f"Stop reason: {result['reason']}")
    print(f"Approximate solution x* ~= {format_vector(result['x_star'])}")
    print(f"lambda* ~= {result['lambda_star']}")
    print(f"f(x*) ~= {result['f_star']}")
    print(f"g(x*) ~= {result['g_star']}")
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
                {"key": "x_k", "title": "x_k"},
                {"key": "lambda_k", "title": "lambda_k", "align": "right"},
                {"key": "f_k", "title": "f(x_k)", "align": "right"},
                {"key": "g_k", "title": "g(x_k)", "align": "right"},
                {"key": "res_norm", "title": "||KKT||_2", "align": "right"},
                {"key": "x_next", "title": "x_{k+1}"},
                {"key": "lambda_next", "title": "lambda_{k+1}", "align": "right"},
                {"key": "f_next", "title": "f(x_{k+1})", "align": "right"},
                {"key": "dx_norm", "title": "||dx||_2", "align": "right"},
                {"key": "dlambda_abs", "title": "|dlambda|", "align": "right"},
                {"key": "decision", "title": "decision"},
            ],
        )
    )
    print()


def main():
    result_a = solve_lagrange_multiplier(
        f_a,
        g_a,
        x0=np.array([0.0, 1.0], dtype=float),
        lambda0=0.0,
        eps1=eps1,
        eps2=eps2,
        delta=delta,
        M=M,
    )
    result_b = solve_lagrange_multiplier(
        f_b,
        g_b,
        x0=np.array([0.0, 0.5], dtype=float),
        lambda0=0.0,
        eps1=eps1,
        eps2=eps2,
        delta=delta,
        M=M,
    )

    print_result("12.a) f(x) = 3x1^2 + 4x2^2 - 2x1x2 + x1,  x1 + x2 = 1", result_a)
    print_result("12.b) f(x) = 5x1^2 + x2^2 - x1x2 + x1,  3x1 + 2x2 = 1", result_b)


if __name__ == "__main__":
    main()
