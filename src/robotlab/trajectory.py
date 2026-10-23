import numpy as np


def quintic(start, end, duration):
    if duration <= 0:
        raise ValueError("positive trajectory duration required")
    start, end = np.asarray(start), np.asarray(end)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("[position,velocity,acceleration] boundary states required")
    coefficients = np.zeros(6)
    coefficients[:3] = [start[0], start[1], start[2] / 2]
    t = duration
    matrix = np.array([[t**3, t**4, t**5], [3 * t**2, 4 * t**3, 5 * t**4], [6 * t, 12 * t**2, 20 * t**3]])
    remaining = end - np.array(
        [
            coefficients[0] + coefficients[1] * t + coefficients[2] * t * t,
            coefficients[1] + 2 * coefficients[2] * t,
            2 * coefficients[2],
        ]
    )
    coefficients[3:] = np.linalg.solve(matrix, remaining)
    return coefficients


def polynomial_state(coefficients, time):
    return np.array(
        [
            np.polynomial.polynomial.polyval(time, np.polynomial.polynomial.polyder(coefficients, k))
            for k in range(3)
        ]
    )


def trapezoid(distance, max_speed, max_acceleration, dt=0.01):
    if distance < 0 or min(max_speed, max_acceleration, dt) <= 0:
        raise ValueError("invalid profile bounds")
    if distance == 0:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])
    ramp = min(max_speed / max_acceleration, np.sqrt(distance / max_acceleration))
    peak = max_acceleration * ramp
    cruise = max(0, (distance - peak * ramp) / peak)
    duration = 2 * ramp + cruise
    time = np.linspace(0, duration, max(2, int(np.ceil(duration / dt)) + 1))
    position = np.empty_like(time)
    speed = np.empty_like(time)
    for k, t in enumerate(time):
        if t < ramp:
            position[k], speed[k] = 0.5 * max_acceleration * t * t, max_acceleration * t
        elif t < ramp + cruise:
            position[k], speed[k] = 0.5 * peak * ramp + peak * (t - ramp), peak
        else:
            remaining = duration - t
            position[k], speed[k] = (
                distance - 0.5 * max_acceleration * remaining**2,
                max_acceleration * remaining,
            )
    return time, position, speed
