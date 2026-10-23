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


def covariance(matrix, size=None):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    if size is not None and matrix.shape != (size, size):
        raise ValueError("unexpected covariance size")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if np.linalg.eigvalsh(matrix).min() < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    return (matrix + matrix.T) / 2


def timeline(duration, dt, events=()):
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("duration and dt must be finite and positive")
    events = np.asarray(events, dtype=float)
    if np.any(~np.isfinite(events)) or np.any((events < 0) | (events > duration)):
        raise ValueError("events outside simulation interval")
    regular = np.arange(int(np.floor(duration / dt)) + 1) * dt
    values = np.unique(np.r_[regular[regular < duration], events, duration])
    return values


def cumulative_integral(time, values):
    time, values = np.asarray(time), np.asarray(values)
    if time.ndim != 1 or values.shape[0] != len(time) or np.any(np.diff(time) <= 0):
        raise ValueError("strictly increasing sample times required")
    scale = np.diff(time).reshape((-1,) + (1,) * (values.ndim - 1))
    intervals = scale * (values[1:] + values[:-1]) / 2
    return np.concatenate([np.zeros_like(values[:1]), np.cumsum(intervals, axis=0)])


def convergence_order(coarse, medium, fine):
    first = np.linalg.norm(np.asarray(coarse) - medium)
    second = np.linalg.norm(np.asarray(medium) - fine)
    if min(first, second) <= 0:
        raise ValueError("distinct nonzero refinement errors required")
    return float(np.log2(first / second))


def bisect(function, low, high, tolerance=1e-10, iterations=100):
    a, b = float(low), float(high)
    fa, fb = function(a), function(b)
    if a >= b or tolerance <= 0 or not np.isfinite(fa + fb) or fa * fb > 0:
        raise ValueError("finite bracket with a sign change required")
    if fa == 0:
        return a
    if fb == 0:
        return b
    for _ in range(iterations):
        middle = (a + b) / 2
        fm = function(middle)
        if not np.isfinite(fm):
            raise ValueError("nonfinite residual")
        if b - a <= tolerance or fm == 0:
            return middle
        if fa * fm <= 0:
            b = middle
        else:
            a, fa = middle, fm
    raise RuntimeError("root iteration limit")
