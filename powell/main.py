import numpy as np

from utils import (
    bracket_minimum_on_ray,
    euclidean_norm,
    format_vector,
    golden_section_phi_search,
    make_cached_nd_function,
    numerical_gradient,
)
from utils.table import format_table


def _minimize_along_direction(eval_f, x, direction, initial_step, max_phi_iterations, phi_eps):
    phi = lambda t: eval_f(x + t * direction)
    f0 = phi(0.0)
    f_plus = phi(initial_step)
    f_minus = phi(-initial_step)

    if f_plus >= f0 and f_minus >= f0:
        return 0.0, x.copy(), f0

    if f_minus < f_plus:
        signed_direction = -direction
        phi = lambda t: eval_f(x + t * signed_direction)
        sign = -1.0
    else:
        signed_direction = direction
        phi = lambda t: eval_f(x + t * signed_direction)
        sign = 1.0

    phi_a, phi_b = bracket_minimum_on_ray(phi, initial_step, max_phi_iterations)
    t_abs, _ = golden_section_phi_search(phi, phi_a, phi_b, phi_eps)
    t_k = sign * t_abs
    x_next = x + t_k * direction
    f_next = eval_f(x_next)
    return t_k, x_next, f_next


def powell_method(
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
    directions = np.eye(x.size, dtype=float)
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
                    "best_dir": "-",
                    "base_steps": "-",
                    "t_disp": "-",
                    "disp_norm": 0.0,
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
                    "best_dir": "-",
                    "base_steps": "-",
                    "t_disp": "-",
                    "disp_norm": 0.0,
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

        x_start = x.copy()
        f_start = fx
        x_curr = x.copy()
        f_curr = fx
        best_dir_index = 0
        best_decrease = -np.inf
        base_step_values = []

        for index, direction in enumerate(directions):
            direction_norm = euclidean_norm(direction)
            if direction_norm == 0:
                base_step_values.append("0")
                continue

            phi_eps = eps2 / max(1.0, direction_norm)
            t_i, x_curr, f_next = _minimize_along_direction(
                eval_f,
                x_curr,
                direction,
                initial_step,
                max_phi_iterations,
                phi_eps,
            )
            decrease = f_curr - f_next
            base_step_values.append(f"{t_i:.6f}")
            if decrease > best_decrease:
                best_decrease = decrease
                best_dir_index = index
            f_curr = f_next

        displacement = x_curr - x_start
        disp_norm = euclidean_norm(displacement)
        t_disp = 0.0
        x_next = x_curr
        f_next = f_curr

        if disp_norm > 0:
            phi_eps = eps2 / max(1.0, disp_norm)
            t_disp, x_next, f_next = _minimize_along_direction(
                eval_f,
                x_curr,
                displacement,
                initial_step,
                max_phi_iterations,
                phi_eps,
            )

            directions[best_dir_index] = displacement

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
                "best_dir": best_dir_index + 1,
                "base_steps": "[" + ", ".join(base_step_values) + "]",
                "t_disp": t_disp,
                "disp_norm": disp_norm,
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
    result = powell_method(f, x0, eps1, eps2, delta, M)

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
                {"key": "best_dir", "title": "best dir", "align": "right"},
                {"key": "base_steps", "title": "t_i"},
                {"key": "t_disp", "title": "t_disp", "align": "right"},
                {"key": "disp_norm", "title": "||disp||_2", "align": "right"},
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
