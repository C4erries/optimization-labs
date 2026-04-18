import numpy as np

from utils import (
    euclidean_norm,
    format_vector,
    make_cached_nd_function,
    numerical_gradient,
    numerical_hessian,
)
from utils.table import format_table


def marquardt_method(
    func,
    x0,
    eps1,
    delta,
    M,
    mu0=20.0,
    max_mu_updates=100,
):
    if eps1 <= 0:
        raise ValueError("eps1 must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")
    if mu0 <= 0:
        raise ValueError("mu0 must be positive.")
    if max_mu_updates <= 0:
        raise ValueError("max_mu_updates must be positive.")

    eval_f, cache, stats = make_cached_nd_function(func)
    x = np.asarray(x0, dtype=float).reshape(-1)
    mu = float(mu0)
    k = 0
    attempt = 0
    history = []

    while True:
        fx = eval_f(x)
        grad = numerical_gradient(eval_f, x, delta)
        grad_norm = euclidean_norm(grad)

        if grad_norm <= eps1:
            history.append(
                {
                    "attempt": "-",
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "step_norm": 0.0,
                    "x_trial": format_vector(x),
                    "f_trial": fx,
                    "mu_next": mu,
                    "decision": "grad <= eps1",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "accepted_iterations": k,
                "trial_steps": attempt,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "gradient",
            }

        if k >= M:
            history.append(
                {
                    "attempt": "-",
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "step_norm": 0.0,
                    "x_trial": format_vector(x),
                    "f_trial": fx,
                    "mu_next": mu,
                    "decision": "k >= M",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "accepted_iterations": k,
                "trial_steps": attempt,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "max_iterations",
            }

        hessian = numerical_hessian(eval_f, x, delta)
        identity = np.eye(x.size, dtype=float)

        for _ in range(max_mu_updates):
            attempt += 1
            modified_hessian = hessian + mu * identity

            try:
                direction = -np.linalg.solve(modified_hessian, grad)
            except np.linalg.LinAlgError:
                next_mu = mu * 2
                history.append(
                    {
                        "attempt": attempt,
                        "k": k,
                        "mu": mu,
                        "x_k": format_vector(x),
                        "f_k": fx,
                        "grad_norm": grad_norm,
                        "step_norm": "-",
                        "x_trial": "-",
                        "f_trial": "-",
                        "mu_next": next_mu,
                        "decision": "solve failed -> mu *= 2",
                    }
                )
                mu = next_mu
                continue

            x_trial = x + direction
            f_trial = eval_f(x_trial)
            step_norm = euclidean_norm(direction)

            if f_trial < fx:
                next_mu = mu / 2
                history.append(
                    {
                        "attempt": attempt,
                        "k": k,
                        "mu": mu,
                        "x_k": format_vector(x),
                        "f_k": fx,
                        "grad_norm": grad_norm,
                        "step_norm": step_norm,
                        "x_trial": format_vector(x_trial),
                        "f_trial": f"{f_trial:.6f}",
                        "mu_next": next_mu,
                        "decision": "accept -> mu /= 2",
                    }
                )
                x = x_trial
                mu = next_mu
                k += 1
                break

            next_mu = mu * 2
            history.append(
                {
                    "attempt": attempt,
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "step_norm": step_norm,
                    "x_trial": format_vector(x_trial),
                    "f_trial": f"{f_trial:.6f}",
                    "mu_next": next_mu,
                    "decision": "reject -> mu *= 2",
                }
            )
            mu = next_mu
        else:
            history.append(
                {
                    "attempt": attempt,
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "step_norm": "-",
                    "x_trial": "-",
                    "f_trial": "-",
                    "mu_next": mu,
                    "decision": "mu update limit",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "accepted_iterations": k,
                "trial_steps": attempt,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "mu_update_limit",
            }


eps1 = 1e-4
delta = 1e-6
M = 100
mu0 = 20.0
x0 = np.array([2.0, 1.5], dtype=float)


def f(x):
    return 3 * x[0] * x[0] + 4 * x[1] * x[1] - 2 * x[0] * x[1] + x[0]


def main():
    result = marquardt_method(f, x0, eps1, delta, M, mu0=mu0)

    print(f"Accepted iterations: {result['accepted_iterations']}")
    print(f"Trial steps: {result['trial_steps']}")
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
                {"key": "attempt", "title": "try", "align": "right"},
                {"key": "k", "title": "k", "align": "right"},
                {"key": "mu", "title": "mu", "align": "right"},
                {"key": "x_k", "title": "x_k"},
                {"key": "f_k", "title": "f(x_k)", "align": "right"},
                {"key": "grad_norm", "title": "||grad||_2", "align": "right"},
                {"key": "step_norm", "title": "||d||_2", "align": "right"},
                {"key": "x_trial", "title": "x_trial"},
                {"key": "f_trial", "title": "f(x_trial)", "align": "right"},
                {"key": "mu_next", "title": "mu_next", "align": "right"},
                {"key": "decision", "title": "decision"},
            ],
        )
    )


if __name__ == "__main__":
    main()
