from math import sqrt

import numpy as np


def format_vector(vector):
    return "[" + ", ".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in vector) + "]"


def infinity_norm(vector):
    array = np.asarray(vector, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def euclidean_norm(vector):
    array = np.asarray(vector, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.linalg.norm(array))


def numerical_gradient(eval_f, x, delta):
    gradient = np.zeros_like(x, dtype=float)

    for index in range(x.size):
        shift = np.zeros_like(x, dtype=float)
        shift[index] = delta
        gradient[index] = (eval_f(x + shift) - eval_f(x - shift)) / (2 * delta)

    return gradient


def numerical_hessian(eval_f, x, delta):
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    hessian = np.zeros((n, n), dtype=float)
    step = float(delta)
    if step <= 0:
        raise ValueError("delta must be positive.")
    fx = eval_f(x)

    for i in range(n):
        shift_i = np.zeros_like(x, dtype=float)
        shift_i[i] = step
        hessian[i, i] = (
            eval_f(x + shift_i) - 2 * fx + eval_f(x - shift_i)
        ) / (step * step)

        for j in range(i + 1, n):
            shift_j = np.zeros_like(x, dtype=float)
            shift_j[j] = step
            value = (
                eval_f(x + shift_i + shift_j)
                - eval_f(x + shift_i - shift_j)
                - eval_f(x - shift_i + shift_j)
                + eval_f(x - shift_i - shift_j)
            ) / (4 * step * step)
            hessian[i, j] = value
            hessian[j, i] = value

    return hessian


def is_positive_definite(matrix):
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        return False

    try:
        np.linalg.cholesky((array + array.T) / 2)
    except np.linalg.LinAlgError:
        return False
    return True


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
