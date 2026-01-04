import numpy as np

def travel_time(v, d):
    """
    Compute seismic travel time through layered medium.

    T(v) = sum(d_i / v_i)
    """
    return np.sum(d / v)


def cost_function(v, d, T_obs, lambda_reg=0.0):
    """
    Regularized least-squares cost function.
    """
    T_calc = travel_time(v, d)
    residual = T_calc - T_obs

    J_data = 0.5 * residual**2
    J_reg = np.sum((v[1:] - v[:-1])**2)

    return J_data + lambda_reg * J_reg


def gradient(v, d, T_obs, lambda_reg=0.0):
    """
    Gradient of the regularized cost function.
    """
    T_calc = travel_time(v, d)
    residual = T_calc - T_obs

    # Data misfit gradient
    grad_data = residual * (-d / v**2)

    # Regularization gradient
    grad_reg = np.zeros_like(v)
    grad_reg[:-1] += 2 * (v[:-1] - v[1:])
    grad_reg[1:]  += 2 * (v[1:] - v[:-1])

    return grad_data + lambda_reg * grad_reg
