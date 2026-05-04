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
    if eps1 <= 0 or delta <= 0 or mu0 <= 0 or max_mu_updates <= 0:
        raise ValueError("eps1, delta, mu0, max_mu_updates must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")

    eval_f, cache, stats = make_cached_nd_function(func)
    x = np.asarray(x0, dtype=float).reshape(-1)
    
    # Шаг 2. Положить k = 0, mu^k = mu^0
    mu = float(mu0)
    k = 0
    total_attempts = 0
    history =[]

    while True:
        # Шаг 3. Вычислить grad f(x^k)
        fx = eval_f(x)
        grad = numerical_gradient(eval_f, x, delta)
        grad_norm = euclidean_norm(grad)

        # Шаг 4. Проверить ||grad|| <= eps1
        if grad_norm <= eps1:
            history.append(
                {
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "attempts": "-",
                    "step_norm": 0.0,
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "decision": "grad <= eps1",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "accepted_iterations": k,
                "trial_steps": total_attempts,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "gradient",
            }

        # Шаг 5. Проверить k >= M
        if k >= M:
            history.append(
                {
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "attempts": "-",
                    "step_norm": 0.0,
                    "x_next": format_vector(x),
                    "f_next": fx,
                    "decision": "k >= M",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "accepted_iterations": k,
                "trial_steps": total_attempts,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "max_iterations",
            }

        # Шаг 6. Вычислить H(x^k)
        hessian = numerical_hessian(eval_f, x, delta)
        identity = np.eye(x.size, dtype=float)
        
        current_attempts = 0

        # Внутренний цикл подбора mu
        for _ in range(max_mu_updates):
            current_attempts += 1
            total_attempts += 1
            
            # Шаг 7. Вычислить H(x^k) + mu * E
            modified_hessian = hessian + mu * identity

            try:
                # Шаг 8 и 9. Вычислить обратную матрицу и d^k 
                # (Мы решаем систему (H + mu*E) * d = -grad)
                direction = -np.linalg.solve(modified_hessian, grad)
            except np.linalg.LinAlgError:
                # Если матрица не положительно определена (или вырождена)
                # Шаг 13: mu = 2*mu и на Шаг 7
                mu *= 2
                continue

            # Шаг 10. Вычислить x^{k+1} = x^k + d^k
            x_trial = x + direction
            f_trial = eval_f(x_trial)
            step_norm = euclidean_norm(direction)

            # Шаг 11. Проверить условие f(x^{k+1}) < f(x^k)
            if f_trial < fx:
                # Шаг 12. Положить k = k+1, mu = mu / 2 и на Шаг 3
                history.append(
                    {
                        "k": k,
                        "mu": mu,
                        "x_k": format_vector(x),
                        "f_k": fx,
                        "grad_norm": grad_norm,
                        "attempts": current_attempts,
                        "step_norm": step_norm,
                        "x_next": format_vector(x_trial),
                        "f_next": f"{f_trial:.6f}",
                        "decision": "accept -> mu/=2",
                    }
                )
                mu /= 2
                x = x_trial
                k += 1
                break
            else:
                # Шаг 13. Положить mu = 2*mu и на Шаг 7 (повторить попытку)
                mu *= 2
        else:
            # Если превышен лимит попыток подбора mu
            history.append(
                {
                    "k": k,
                    "mu": mu,
                    "x_k": format_vector(x),
                    "f_k": fx,
                    "grad_norm": grad_norm,
                    "attempts": current_attempts,
                    "step_norm": "-",
                    "x_next": "-",
                    "f_next": "-",
                    "decision": "mu update limit",
                }
            )
            return {
                "x_star": x,
                "f_star": fx,
                "accepted_iterations": k,
                "trial_steps": total_attempts,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "mu_update_limit",
            }


eps1 = 1e-4
# delta = 1e-4 для численного Гессиана
delta = 1e-4
M = 100
mu0 = 20.0
x0 = np.array([2.0, 1.5], dtype=float)


def f(x):
    return 3 * x[0] * x[0] + 4 * x[1] * x[1] - 2 * x[0] * x[1] + x[0]


def main():
    result = marquardt_method(f, x0, eps1, delta, M, mu0=mu0)

    print(f"Accepted iterations: {result['accepted_iterations']}")
    print(f"Total inner trial steps: {result['trial_steps']}")
    print(f"Stop reason: {result['reason']}")
    print(f"Approximate solution x* ~= {format_vector(result['x_star'])}")
    print(f"f(x*) ~= {result['f_star']}")
    print(
        "Function evaluations:",
        f"total requests = {result['stats']['requests']},",
        f"computed(N) = {result['stats']['computed']},",
    )
    print("\nSteps:")
    print(
        format_table(
            result["history"],[
                {"key": "k", "title": "k", "align": "right"},
                {"key": "x_k", "title": "x_k"},
                {"key": "f_k", "title": "f(x_k)", "align": "right"},
                {"key": "grad_norm", "title": "||grad||_2", "align": "right"},
                {"key": "mu", "title": "mu", "align": "right"},
                {"key": "attempts", "title": "Inner tries", "align": "right"},
                {"key": "step_norm", "title": "||d||_2", "align": "right"},
                {"key": "x_next", "title": "x_{k+1}"},
                {"key": "f_next", "title": "f(x_{k+1})", "align": "right"},
                {"key": "decision", "title": "decision"},
            ],
        )
    )


if __name__ == "__main__":
    main()