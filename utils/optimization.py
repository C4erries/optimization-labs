from math import sqrt

import numpy as np


def format_vector(vector):
    return "[" + ", ".join(f"{value:.6f}".rstrip("0").rstrip(".") for value in vector) + "]"


def infinity_norm(vector):
    array = np.asarray(vector, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def numerical_gradient(eval_f, x, delta):
    gradient = np.zeros_like(x, dtype=float)

    for index in range(x.size):
        shift = np.zeros_like(x, dtype=float)
        shift[index] = delta
        gradient[index] = (eval_f(x + shift) - eval_f(x - shift)) / (2 * delta)

    return gradient


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

    raise RuntimeError("Failed to bracket a line-search minimum.")


def golden_section_line_search(phi, a, b, length_limit):
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
