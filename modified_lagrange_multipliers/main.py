import numpy as np

from davidson_fletcher_powell.main import davidson_fletcher_powell
from utils import euclidean_norm, format_vector, make_cached_nd_function
from utils.table import format_table


def _as_constraint_list(constraints):
    if callable(constraints):
        return[constraints]
    return list(constraints)


def modified_lagrange_multipliers(
    func,
    constraints,           # Ограничения типа РАВЕНСТВ: g(x) = 0
    x0,
    eps1,
    delta,
    M,
    r0=1.0,
    c=10.0,
    inner_M=100,
    ineq_constraints=None, # Ограничения типа НЕРАВЕНСТВ: g(x) <= 0
    lambda0=None,          # Множители для равенств
    mu0=None,              # Множители для неравенств
):
    if eps1 <= 0 or delta <= 0 or r0 <= 0:
        raise ValueError("eps1, delta, r0 must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")
    if c <= 1:
        raise ValueError("c must be greater than 1.")

    eval_f, cache_f, stats_f = make_cached_nd_function(func)
    
    # Обработка равенств
    constraint_functions = _as_constraint_list(constraints) if constraints else []
    cached_eq =[make_cached_nd_function(g) for g in constraint_functions]
    eval_eq =[item[0] for item in cached_eq]
    
    # Обработка неравенств
    ineq_funcs = _as_constraint_list(ineq_constraints) if ineq_constraints else[]
    cached_ineq =[make_cached_nd_function(g) for g in ineq_funcs]
    eval_ineq = [item[0] for item in cached_ineq]

    x = np.asarray(x0, dtype=float).reshape(-1)
    
    lambdas = np.asarray(lambda0, dtype=float).reshape(-1) if lambda0 is not None else np.zeros(len(eval_eq))
    mus = np.asarray(mu0, dtype=float).reshape(-1) if mu0 is not None else np.zeros(len(eval_ineq))

    if lambdas.size != len(eval_eq):
        raise ValueError("lambda0 length must match equality constraints count.")
    if mus.size != len(eval_ineq):
        raise ValueError("mu0 length must match inequality constraints count.")

    r = float(r0)
    history =[]
    k = 0

    while True:
        x_k = x.copy()
        f_k = eval_f(x_k)

        # Шаг 2. Составить модифицированную функцию Лагранжа L(x, lambda, mu, r)
        def P_func(point, current_lambdas, current_mus, current_r):
            # Вычисляет P(x, mu, r) = штрафная часть Лагранжиана (Шаг 4)
            p_val = 0.0
            if eval_eq:
                eq_vals = np.array([g(point) for g in eval_eq], dtype=float)
                p_val += (current_r / 2.0) * np.sum(eq_vals**2)
            if eval_ineq:
                ineq_vals = np.array([g(point) for g in eval_ineq], dtype=float)
                # По формуле Шага 4: 1/(2r) * sum( max{0, mu + r*g}^2 - mu^2 )
                max_term = np.maximum(0.0, current_mus + current_r * ineq_vals)
                p_val += (1.0 / (2.0 * current_r)) * np.sum(max_term**2 - current_mus**2)
            return p_val

        def augmented_lagrangian(point):
            val = eval_f(point)
            if eval_eq:
                eq_vals = np.array([g(point) for g in eval_eq], dtype=float)
                val += np.dot(lambdas, eq_vals)
            val += P_func(point, lambdas, mus, r)
            return val

        # Шаг 3. Найти точку x* безусловного минимума
        inner = davidson_fletcher_powell(
            augmented_lagrangian,
            x_k,
            eps1,
            # eps2 - убрали, так как в методе множителей критерий остановки другой
            eps1, 
            delta,
            inner_M,
        )

        x_next = inner["x_star"]
        f_next = eval_f(x_next)
        
        eq_vals_next = np.array([g(x_next) for g in eval_eq], dtype=float) if eval_eq else np.array([])
        ineq_vals_next = np.array([g(x_next) for g in eval_ineq], dtype=float) if eval_ineq else np.array([])
        
        # Шаг 4. Вычислить P(x*, mu^k, r^k)
        penalty_val = P_func(x_next, lambdas, mus, r)
        
        # Пересчет множителей (Шаг 4.б)
        lambdas_next = lambdas + r * eq_vals_next if eval_eq else lambdas
        mus_next = np.maximum(0.0, mus + r * ineq_vals_next) if eval_ineq else mus

        dx_norm = euclidean_norm(x_next - x_k)

        # а) если P <= eps1, процесс поиска закончить
        stopped = False
        reason = ""
        if abs(penalty_val) <= eps1:
            decision = "P <= eps1 -> stop"
            stopped = True
            reason = "penalty_converged"
        elif k >= M:
            decision = "k >= M -> stop"
            stopped = True
            reason = "max_iterations"
        else:
            # б) если P > eps1: пересчет множителей и параметра штрафа
            decision = "P > eps1 -> r *= C, update multipliers"

        history.append(
            {
                "k": k,
                "r_k": r,
                "x_k": format_vector(x_k),
                "lambdas": format_vector(lambdas) if eval_eq else "-",
                "mus": format_vector(mus) if eval_ineq else "-",
                "f_k": f_k,
                "penalty": penalty_val,
                "inner_iter": inner["iterations"],
                "x_next": format_vector(x_next),
                "f_next": f"{f_next:.6f}",
                "dx_norm": dx_norm,
                "decision": decision,
            }
        )

        if stopped:
            return _make_result(
                x_next, lambdas_next, mus_next, f_next, 
                eq_vals_next, ineq_vals_next, 
                history, cache_f, stats_f, 
                cached_eq + cached_ineq, reason
            )

        x = x_next
        lambdas = lambdas_next
        mus = mus_next
        r *= c
        k += 1


def _make_result(x, lambdas, mus, fx, eq_vals, ineq_vals, history, cache_f, stats_f, cached_constraints, reason):
    return {
        "x_star": x,
        "lambda_star": lambdas,
        "mu_star": mus,
        "f_star": fx,
        "g_eq_star": eq_vals,
        "g_ineq_star": ineq_vals,
        "iterations": len(history),
        "history": history,
        "cache": {
            "f": cache_f,
            "g":[item[1] for item in cached_constraints],
        },
        "stats": {
            "f_requests": stats_f["requests"],
            "f_computed": stats_f["computed"],
            "g_requests": sum(item[2]["requests"] for item in cached_constraints),
            "g_computed": sum(item[2]["computed"] for item in cached_constraints),
        },
        "reason": reason,
    }


eps1 = 0.001
delta = 1e-6
M = 20

def f(x):
    # f(x) = x^2 - 4x -> min
    return x[0]**2 - 4*x[0]

def g_ineq(x):
    # g1(x) = x - 1 <= 0
    return x[0] - 1


def main():
    result = modified_lagrange_multipliers(
        func=f,
        constraints=None,
        ineq_constraints=g_ineq,
        x0=np.array([0.0], dtype=float),
        eps1=eps1,
        delta=delta,
        M=M,
        r0=1.0,
        c=10.0,
        mu0=np.array([0.0], dtype=float),
    )

    print(f"Iterations: {result['iterations']}")
    print(f"Stop reason: {result['reason']}")
    print(f"Approximate solution x* ~= {format_vector(result['x_star'])}")
    print(f"mu* (inequality multipliers) ~= {format_vector(result['mu_star'])}")
    print(f"f(x*) ~= {result['f_star']}")
    
    print("\nSteps:")
    print(
        format_table(
            result["history"],[
                {"key": "k", "title": "k", "align": "right"},
                {"key": "r_k", "title": "r_k", "align": "right"},
                {"key": "mus", "title": "mu_k"},
                {"key": "x_next", "title": "x_{k+1}"},
                {"key": "f_next", "title": "f(x_{k+1})", "align": "right"},
                {"key": "penalty", "title": "P(x_{k+1})", "align": "right"},
                {"key": "inner_iter", "title": "inner", "align": "right"},
                {"key": "decision", "title": "decision"},
            ],
        )
    )

if __name__ == "__main__":
    main()