import numpy as np

from utils import (
    bracket_minimum_on_ray,
    euclidean_norm,
    format_vector,
    golden_section_phi_search,
    is_positive_definite,
    make_cached_nd_function,
    numerical_gradient,
    numerical_hessian,
)
from utils.table import format_table


def newton_raphson(
    func,
    x0,
    eps1,
    eps2,
    delta,
    M,
    initial_step=0.1,
    max_phi_iterations=50,
):
    if eps1 <= 0:
        raise ValueError("eps1 must be positive.")
    if eps2 <= 0:
        raise ValueError("eps2 must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")

    eval_f, cache, stats = make_cached_nd_function(func)
    x = np.asarray(x0, dtype=float).reshape(-1)
    history = []
    k = 0
    prev_eps2_satisfied = False

    while True:
        fx = eval_f(x)
        grad = numerical_gradient(eval_f, x, delta)
        grad_norm = euclidean_norm(grad)

        if grad_norm <= eps1:
            history.append(
                {
                    "k": k,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "direction_type": "-",
                    "t_k": "-",
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "df_abs": 0.0,
                    "decision": "grad <= eps1",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "iterations": len(history),
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "gradient",
            }

        if k >= M:
            history.append(
                {
                    "k": k,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "direction_type": "-",
                    "t_k": "-",
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "df_abs": 0.0,
                    "decision": "k >= M",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "iterations": len(history),
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "max_iterations",
            }

        hessian = numerical_hessian(eval_f, x, delta)

        if is_positive_definite(hessian):
            try:
                direction = -np.linalg.solve(hessian, grad)
                direction_type = "Newton"
            except np.linalg.LinAlgError:
                direction = -grad
                direction_type = "Steepest fallback"
        else:
            direction = -grad
            direction_type = "Steepest fallback"

        direction_norm = euclidean_norm(direction)
        if direction_norm == 0:
            history.append(
                {
                    "k": k,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "direction_type": direction_type,
                    "t_k": "-",
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "dx_norm": 0.0,
                    "df_abs": 0.0,
                    "decision": "zero direction",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "iterations": len(history),
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "zero_direction",
            }

        phi = lambda t: eval_f(x + t * direction)
        phi_a, phi_b = bracket_minimum_on_ray(phi, initial_step, max_phi_iterations)
        phi_eps = eps2 / max(1.0, direction_norm)
        t_k, _ = golden_section_phi_search(phi, phi_a, phi_b, phi_eps)

        x_next = x + t_k * direction
        f_next = eval_f(x_next)
        dx_norm = euclidean_norm(x_next - x)
        df_abs = abs(f_next - fx)

        decision = "continue"
        eps2_satisfied = dx_norm <= eps2 and df_abs <= eps2
        if eps2_satisfied and prev_eps2_satisfied:
            decision = "eps2 x2 -> stop"
        elif eps2_satisfied:
            decision = "eps2 x1"

        history.append(
            {
                "k": k,
                "x_k": format_vector(x),
                "f_k": fx,
                "grad_norm": grad_norm,
                "direction_type": direction_type,
                "t_k": t_k,
                "x_next": format_vector(x_next),
                "f_next": f"{f_next:.6f}",
                "dx_norm": dx_norm,
                "df_abs": df_abs,
                "decision": decision,
            }
        )

        if eps2_satisfied and prev_eps2_satisfied:
            return {
                "x_star": x_next,
                "f_star": f_next,
                "iterations": len(history),
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "point_and_value_twice",
            }

        prev_eps2_satisfied = eps2_satisfied
        x = x_next
        k += 1


eps1 = 1e-4
eps2 = 1e-4
delta = 1e-6
M = 100
x0 = np.array([2.0, 1.5], dtype=float)


def f(x):
    return 3 * x[0] * x[0] + 4 * x[1] * x[1] - 2 * x[0] * x[1] + x[0]


def main():
    result = newton_raphson(f, x0, eps1, eps2, delta, M)

    print(f"Iterations: {result['iterations']}")
    print(f"Stop reason: {result['reason']}")
    print(f"Approximate solution x* ~= {format_vector(result['x_star'])}")
    print(f"f(x*) ~= {result['f_star']}")
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
                {"key": "x_k", "title": "x_k"},
                {"key": "f_k", "title": "f(x_k)", "align": "right"},
                {"key": "grad_norm", "title": "||grad||_2", "align": "right"},
                {"key": "direction_type", "title": "direction", "align": "right"},
                {"key": "t_k", "title": "t_k", "align": "right"},
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
