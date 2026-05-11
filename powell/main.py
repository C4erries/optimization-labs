import numpy as np
from math import sqrt

# Оставляем импорты для таблиц и кэширования из твоей структуры
from utils import (
    make_cached_nd_function,
)
from utils.table import format_table


def format_vector(vector):
    return "[" + ", ".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in vector) + "]"

def euclidean_norm(vector):
    array = np.asarray(vector, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array))

def bracket_minimum_on_ray(phi, initial_step, max_iterations):
    if initial_step <= 0:
        raise ValueError("initial_step must be positive.")

    t_prev = 0.0
    f_prev = phi(t_prev)
    t_curr = initial_step
    f_curr = phi(t_curr)

    if f_curr >= f_prev:
        return t_prev, t_curr

    current_step = initial_step

    for _ in range(max_iterations):
        current_step *= 2
        t_next = current_step
        f_next = phi(t_next)

        if f_next >= f_curr:
            return t_prev, t_next

        t_prev = t_curr
        f_prev = f_curr
        t_curr = t_next
        f_curr = f_next

    raise RuntimeError("Failed to bracket a phi minimum.")

def golden_section_phi_search(phi, a, b, length_limit):
    if b <= a:
        raise ValueError("Right border must be greater than left border.")
    if length_limit <= 0:
        raise ValueError("length_limit must be positive.")

    tau = (sqrt(5) - 1) / 2
    x1 = b - tau * (b - a)
    x2 = a + tau * (b - a)
    fx1 = phi(x1)
    fx2 = phi(x2)

    while b - a > length_limit:
        if fx1 <= fx2:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = b + a - x2
            fx1 = phi(x1)
        else:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = b + a - x1
            fx2 = phi(x2)

    return (a + b) / 2, (a, b)


def powell_method(
    func,
    x0,
    eps,
    M,
    initial_step=0.1,
    max_phi_iterations=50,
):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if M < 0:
        raise ValueError("M must be non-negative.")

    eval_f, cache, stats = make_cached_nd_function(func)
    
    xk = np.asarray(x0, dtype=float).reshape(-1)
    n = len(xk)
    x_star = xk.copy()
    
    d = np.eye(n)
    d = np.vstack([d[-1], d])  # Матрица направлений
    
    yi = xk.copy()
    y0 = yi.copy()
    
    history = []
    k = 0

    while k < M:
        t_values = []
        stop_reason = None
        
        # Внутренний цикл по направлениям
        for i in range(n + 1):
            direction = d[i]
            
            if euclidean_norm(direction) == 0:
                t = 0.0
            else:
                phi = lambda t: eval_f(yi + t * direction)
                
                f0 = phi(0.0)
                f_plus = phi(initial_step)
                f_minus = phi(-initial_step)
                
                if f_plus >= f0 and f_minus >= f0:
                    t = 0.0
                else:
                    sign = -1.0 if f_minus < f_plus else 1.0
                    phi_directed = lambda step: eval_f(yi + step * sign * direction)
                    
                    a, b = bracket_minimum_on_ray(phi_directed, initial_step, max_phi_iterations)
                    
                    t_abs, _ = golden_section_phi_search(phi_directed, a, b, eps)
                    t = sign * t_abs
            
            t_values.append(t)
            
            yi = yi + t * direction
            
            if i == 0:
                y1 = yi.copy()
                
            # Проверки условий остановки внутри цикл
            if (i == n - 1) and (euclidean_norm(yi - y0) < eps):
                x_star = yi.copy()
                stop_reason = "y^n == y^0"
                break
                
            if (i == n) and (euclidean_norm(yi - y1) < eps):
                x_star = yi.copy()
                stop_reason = "y^{n+1} == y^1"
                break

        if stop_reason:
            t_str = ", ".join(f"t{idx}={val:.4f}" for idx, val in enumerate(t_values))
            history.append({
                "k": k, "x_k": format_vector(xk), "t_values": t_str,
                "new_dir": "-", "rank": "-", "x_next": format_vector(x_star),
                "dx_norm": 0.0, "decision": f"stop ({stop_reason})"
            })
            return {"x_star": x_star, "f_star": eval_f(x_star), "iterations": k + 1, "history": history, "cache": cache, "stats": stats, "reason": stop_reason}
            
        x_next = yi.copy()
        dx_norm = euclidean_norm(x_next - xk)
        
        if dx_norm < eps:
            x_star = x_next.copy()
            stop_reason = "dx < eps"
            t_str = ", ".join(f"t{idx}={val:.4f}" for idx, val in enumerate(t_values))
            history.append({
                "k": k, "x_k": format_vector(xk), "t_values": t_str,
                "new_dir": "-", "rank": "-", "x_next": format_vector(x_star),
                "dx_norm": f"{dx_norm:.6f}", "decision": f"stop ({stop_reason})"
            })
            return {"x_star": x_star, "f_star": eval_f(x_star), "iterations": k + 1, "history": history, "cache": cache, "stats": stats, "reason": stop_reason}

        # Обновление матрицы направлений
        new_dir = yi - y1
        d_ = np.delete(d, 1, axis=0)
        d_[0] = new_dir.copy()
        d_ = np.vstack([d_, new_dir.copy()])
        
        rank = np.linalg.matrix_rank(d_[1:])
        
        if rank == n:
            d = d_
            y0 = x_next.copy()
            decision = "update dirs"
        else:
            y0 = x_next.copy()
            decision = "keep old dirs"
            
        t_str = ", ".join(f"t{idx}={val:.4f}" for idx, val in enumerate(t_values))
        history.append({
            "k": k, "x_k": format_vector(xk), "t_values": t_str,
            "new_dir": format_vector(new_dir), "rank": f"{rank}/{n}",
            "x_next": format_vector(x_next), "dx_norm": f"{dx_norm:.6f}", "decision": decision
        })
        
        xk = x_next.copy()
        k += 1

    return {"x_star": xk, "f_star": eval_f(xk), "iterations": k, "history": history, "cache": cache, "stats": stats, "reason": "max_iterations"}


eps = 1e-5
M = 100
x0 = np.array([20.0, -10.5], dtype=float)

def f(x):
    return 3 * x[0] * x[0] + 4 * x[1] * x[1] - 2 * x[0] * x[1] + x[0]

def main():
    result = powell_method(f, x0, eps, M)

    print(f"Iterations: {result['iterations']}")
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
            result["history"], [
                {"key": "k", "title": "k", "align": "right"},
                {"key": "x_k", "title": "x_k"},
                {"key": "t_values", "title": "1D steps (t_i)"}, 
                {"key": "new_dir", "title": "new d_n (y^{n+1}-y^1)"},
                {"key": "rank", "title": "rank"},
                {"key": "x_next", "title": "x_{k+1}"},
                {"key": "dx_norm", "title": "||dx||_2", "align": "right"},
                {"key": "decision", "title": "decision"},
            ],
        )
    )

if __name__ == "__main__":
    main()