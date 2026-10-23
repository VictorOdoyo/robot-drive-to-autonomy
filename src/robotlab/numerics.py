import numpy as np
from scipy.linalg import expm


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


def discretize(matrix, input_matrix, dt):
    matrix, input_matrix = np.asarray(matrix), np.asarray(input_matrix)
    n, k = input_matrix.shape
    if matrix.shape != (n, n) or dt < 0:
        raise ValueError("incompatible state space or timestep")
    block = np.zeros((n + k, n + k))
    block[:n, :n], block[:n, n:] = matrix, input_matrix
    transition = expm(block * dt)
    return transition[:n, :n], transition[:n, n:]


def process_noise(matrix, spectral_density, dt):
    matrix, spectral_density = np.asarray(matrix), np.asarray(spectral_density)
    n = len(matrix)
    if dt < 0 or spectral_density.shape != (n, n):
        raise ValueError("invalid process noise discretization")
    block = np.block([[matrix, spectral_density], [np.zeros_like(matrix), -matrix.T]])
    value = expm(block * dt)
    covariance = value[:n, n:] @ value[:n, :n].T
    return (covariance + covariance.T) / 2
