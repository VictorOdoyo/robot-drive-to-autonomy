import numpy as np


def jacobian(function, point, epsilon=1e-6):
    point = np.asarray(point, dtype=float)
    columns = []
    for index in range(len(point)):
        delta = np.zeros_like(point)
        delta[index] = epsilon * max(1, abs(point[index]))
        columns.append((np.asarray(function(point + delta)) - function(point - delta)) / (2 * delta[index]))
    return np.column_stack(columns)


def rk4(function, time, state, dt):
    if dt <= 0:
        raise ValueError("dt must be positive")
    state = np.asarray(state, dtype=float)
    k1 = function(time, state)
    k2 = function(time + dt / 2, state + dt * np.asarray(k1) / 2)
    k3 = function(time + dt / 2, state + dt * np.asarray(k2) / 2)
    k4 = function(time + dt, state + dt * np.asarray(k3))
    return state + dt * (np.asarray(k1) + 2 * np.asarray(k2) + 2 * np.asarray(k3) + np.asarray(k4)) / 6
