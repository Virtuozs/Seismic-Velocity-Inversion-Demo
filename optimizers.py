import numpy as np
from models import gradient, cost_function

def gradient_descent(
    v0,
    d,
    T_obs,
    learning_rate,
    iterations,
    lambda_reg=0.0
):
    """
    Standard Gradient Descent optimization.
    """
    v = v0.copy()
    history = []

    for k in range(iterations):
        J = cost_function(v, d, T_obs, lambda_reg)
        history.append(J)

        grad = gradient(v, d, T_obs, lambda_reg)
        v = v - learning_rate * grad

    return v, history

def quasi_newton(
    v0,
    d,
    T_obs,
    learning_rate,
    iterations,
    lambda_reg=0.0
):
    """
    Quasi-Newton optimization using diagonal inverse Hessian.
    """
    v = v0.copy()
    history = []

    for k in range(iterations):
        J = cost_function(v, d, T_obs, lambda_reg)
        history.append(J)

        grad = gradient(v, d, T_obs, lambda_reg)

        # Diagonal inverse Hessian approximation
        H_inv = (v**4) / (d**2)

        v = v - learning_rate * H_inv * grad

    return v, history
