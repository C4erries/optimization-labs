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
    constraints,           # Ограничения типа РАВЕНСТВ: g(x) = 0
    x0,
    eps1,
    eps2,
    delta,
    M,
    r0=1.0,
    c=10.0,
    inner_M=100,
    ineq_constraints=None, # Ограничения типа НЕРАВЕНСТВ: g(x) <= 0
):
    if eps1 <= 0 or eps2 <= 0 or delta <= 0 or r0 <= 0:
        raise ValueError("eps1, eps2, delta, r0 must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")
    if c <= 1:
        raise ValueError("c must be greater than 1.")

    eval_f, cache_f, stats_f = make_cached_nd_function(func)
    
    # Кешируем функции-равенства
    constraint_functions = _as_constraint_list(constraints) if constraints else []
    cached_eq =[make_cached_nd_function(g) for g in constraint_functions]
    eval_eq = [item[0] for item in cached_eq]
    
    # Кешируем функции-неравенства (срезки)
    if ineq_constraints is not None:
        ineq_funcs = _as_constraint_list(ineq_constraints)
        cached_ineq =[make_cached_nd_function(g) for g in ineq_funcs]
        eval_ineq = [item[0] for item in cached_ineq]
    else:
        cached_ineq =[]
        eval_ineq =[]

    # Шаг 1. Задать начальную точку x^0, r^0 > 0, C > 1, eps > 0, k = 0
    x = np.asarray(x0, dtype=float).reshape(-1)
    r = float(r0)
    history =[]
    k = 0

    while True:
        # Начало k-й итерации
        x_k = x.copy()
        f_k = eval_f(x_k)

        # Шаг 2. Составить вспомогательную функцию F(x, r^k) = f(x) + P(x, r^k)
        def P_func(point, r_val):
            p_val = 0.0
            # Квадратичный штраф для равенств
            if eval_eq:
                eq_vals = np.array([g(point) for g in eval_eq], dtype=float)
                p_val += np.sum(eq_vals**2)
            # Штраф "квадрат срезки" для неравенств
            if eval_ineq:
                ineq_vals = np.array([max(0.0, g(point)) for g in eval_ineq], dtype=float)
                p_val += np.sum(ineq_vals**2)
            return 0.5 * r_val * p_val

        def F_func(point):
            return eval_f(point) + P_func(point, r)

        # Шаг 3. Найти точку x*(r^k) безусловного минимума с помощью DFP
        inner = davidson_fletcher_powell(
            F_func,
            x_k,
            eps1,
            eps2,
            delta,
            inner_M,
        )

        x_next = inner["x_star"]
        f_next = eval_f(x_next)
        
        # Вычисляем штраф в найденной точке: P(x*(r^k), r^k)
        P_next = P_func(x_next, r)
        
        # Считаем норму нарушения ограничений для вывода в таблицу
        eq_vals_next = np.array([g(x_next) for g in eval_eq], dtype=float) if eval_eq else np.array([])
        ineq_vals_next = np.array([g(x_next) for g in eval_ineq], dtype=float) if eval_ineq else np.array([])
        
        total_violation = 0.0
        if eval_eq:
            total_violation += np.sum(eq_vals_next**2)
        if eval_ineq:
            ineq_viol = np.array([max(0.0, v) for v in ineq_vals_next])
            total_violation += np.sum(ineq_viol**2)
        total_viol_norm = np.sqrt(total_violation)

        dx_norm = euclidean_norm(x_next - x_k)
        df_abs = abs(f_next - f_k)

        # Шаг 4. Проверить условие окончания
        stopped = False
        reason = ""
        
        # а) если P(x*(r^k), r^k) <= eps1, процесс поиска закончить
        if P_next <= eps1:
            decision = "P <= eps1 -> stop"
            stopped = True
            reason = "penalty_converged"
        elif k >= M:
            decision = "k >= M -> stop"
            stopped = True
            reason = "max_iterations"
        else:
            # б) если P > eps1, положить r^{k+1} = C * r^k, перейти к шагу 2
            decision = "P > eps1 -> r *= c"

        history.append(
            {
                "k": k,
                "r_k": r,
                "x_k": format_vector(x_k),
                "f_k": f_k,
                "g_norm": total_viol_norm,
                "penalty": P_next,
                "inner_iter": inner["iterations"],
                "x_next": format_vector(x_next),
                "f_next": f"{f_next:.6f}",
                "dx_norm": dx_norm,
                "df_abs": df_abs,
                "decision": decision,
            }
        )

        if stopped:
            return _make_result(
                x_next,
                f_next,
                eq_vals_next, # возвращаем значения равенств как результат
                history,
                cache_f,
                stats_f,
                cached_eq + cached_ineq,
                reason,
            )

        # Подготовка к следующей итерации (Шаг 4.б)
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
    print("\nSteps:")
    print(
        format_table(
            result["history"],[
                {"key": "k", "title": "k", "align": "right"},
                {"key": "r_k", "title": "r_k", "align": "right"},
                {"key": "x_k", "title": "x_k"},
                {"key": "f_k", "title": "f(x_k)", "align": "right"},
                {"key": "g_norm", "title": "||g(x_{k+1})||", "align": "right"},
                {"key": "penalty", "title": "P(x_{k+1})", "align": "right"},
                {"key": "inner_iter", "title": "inner_iters", "align": "right"},
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