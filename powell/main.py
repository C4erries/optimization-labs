import numpy as np

from utils import (
    bracket_minimum_on_ray,
    euclidean_norm,
    format_vector,
    golden_section_phi_search,
    make_cached_nd_function,
)
from utils.table import format_table


def _minimize_along_direction(eval_f, x, direction, initial_step, max_phi_iterations, phi_eps):
    # Эта функция у вас написана верно: она делает 1D минимизацию вдоль луча
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
    x = np.asarray(x0, dtype=float).reshape(-1)
    n = x.size
    
    # Шаг 1. Задать начальные направления поиска (единичные орты)
    # Используем индексацию 1..n для удобства сопоставления с учебником.
    # d[0] будет использоваться для хранения d_n.
    d =[np.zeros(n) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][i - 1] = 1.0

    history =[]
    k = 0
    x_k = x.copy()

    while k < M:
        # Шаг 1 (начало итерации): Положим d_0 = d_n, y^0 = x^k
        d[0] = d[n].copy()
        
        # Массив точек y^0, y^1, ..., y^{n+1}
        y =[np.zeros(n) for _ in range(n + 2)]
        y[0] = x_k.copy()
        
        # Массив для хранения скалярных шагов t_i
        t = np.zeros(n + 1)
        
        stop_reason = None
        
        # Шаг 2. Найти y^{i+1} = y^i + t_i * d_i для i = 0, ..., n
        for i in range(n + 1):
            direction = d[i]
            dir_norm = euclidean_norm(direction)
            
            if dir_norm == 0:
                t[i] = 0.0
                y[i+1] = y[i].copy()
            else:
                phi_eps = eps / max(1.0, dir_norm)
                t[i], y[i+1], _ = _minimize_along_direction(
                    eval_f,
                    y[i],
                    direction,
                    initial_step,
                    max_phi_iterations,
                    phi_eps,
                )
                
            # Шаг 3. Проверить выполнение условий остановки внутри цикла
            if i == n - 1:
                # б) если i = n - 1, проверить y^n == y^0
                if euclidean_norm(y[n] - y[0]) < 1e-10:
                    x_k = y[n].copy()
                    stop_reason = "y^n == y^0"
                    break
            elif i == n:
                # в) если i = n, проверить y^{n+1} == y^1
                if euclidean_norm(y[n+1] - y[1]) < 1e-10:
                    x_k = y[n+1].copy()
                    stop_reason = "y^{n+1} == y^1"
                    break

        if stop_reason:
            t_str = ", ".join(f"t{idx}={val:.4f}" for idx, val in enumerate(t[:i+1]))
            history.append({
                "k": k,
                "x_k": format_vector(y[0]),
                "t_values": t_str,
                "new_dir": "-",
                "rank": "-",
                "x_next": format_vector(x_k),
                "dx_norm": 0.0,
                "decision": f"stop ({stop_reason})"
            })
            return {
                "x_star": x_k,
                "f_star": eval_f(x_k),
                "iterations": k + 1,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": stop_reason,
            }
            
        # Шаг 4. Положить x^{k+1} = y^{n+1}
        x_next = y[n+1].copy()
        dx_norm = euclidean_norm(x_next - y[0])
        
        # Строим новое сопряженное направление
        d_new = y[n+1] - y[1]
        
        # Формируем новую систему направлений: d_bar
        d_bar =[np.zeros(n) for _ in range(n + 1)]
        d_bar[n] = d_new.copy()
        for i in range(1, n):
            d_bar[i] = d[i+1].copy()
            
        # Проверяем линейную независимость (ранг)
        matrix_D = np.column_stack([d_bar[i] for i in range(1, n+1)])
        rank = np.linalg.matrix_rank(matrix_D)
        
        if dx_norm < eps:
            decision = "stop (||dx|| < eps)"
        else:
            decision = "update dirs" if rank == n else "keep old dirs"
            
        t_str = ", ".join(f"t{idx}={val:.4f}" for idx, val in enumerate(t))
        
        history.append({
            "k": k,
            "x_k": format_vector(y[0]),
            "t_values": t_str, # Выводим t_i как независимые скаляры!
            "new_dir": format_vector(d_new),
            "rank": f"{rank}/{n}",
            "x_next": format_vector(x_next),
            "dx_norm": f"{dx_norm:.6f}",
            "decision": decision
        })
        
        # a) если ||x^{k+1} - x^k|| < eps, поиск завершить
        if dx_norm < eps:
            return {
                "x_star": x_next,
                "f_star": eval_f(x_next),
                "iterations": k + 1,
                "history": history,
                "cache": cache,
                "stats": stats,
                "reason": "dx < eps",
            }
            
        # Обновляем направления только если матрица независима
        if rank == n:
            for i in range(1, n+1):
                d[i] = d_bar[i].copy()
        # Иначе - оставляем старые
        
        x_k = x_next
        k += 1

    return {
        "x_star": x_k,
        "f_star": eval_f(x_k),
        "iterations": k,
        "history": history,
        "cache": cache,
        "stats": stats,
        "reason": "max_iterations",
    }


eps = 1e-4
M = 100
x0 = np.array([
    2.0, 
    1.5, 
    # 0
    ], dtype=float)
# (x[0]+2*x[1]-5)**4 + (x[1]-x[2])**2 + 3 + (x[0]+x[1]+x[2]-7)**2

def f(x):
    # return (x[0]+2*x[1]-5)**4 + (x[1]-x[2])**2 + 3 + (x[0]+x[1]+x[2]-7)**2
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
            result["history"],[
                {"key": "k", "title": "k", "align": "right"},
                {"key": "x_k", "title": "x_k"},
                # Заменил "векторный" вывод t_i на явный скалярный:
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

