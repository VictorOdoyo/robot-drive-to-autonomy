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


def arc_length(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("at least two planar waypoints required")
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def resample_path(points, spacing=0.05):
    if spacing <= 0:
        raise ValueError("positive path spacing required")
    points = np.asarray(points)
    distance = arc_length(points)
    keep = np.r_[True, np.diff(distance) > 1e-12]
    if np.sum(keep) < 2:
        raise ValueError("path has zero length")
    samples = np.linspace(0, distance[-1], max(2, int(np.ceil(distance[-1] / spacing)) + 1))
    return np.column_stack([np.interp(samples, distance[keep], points[keep, k]) for k in range(2)])


def path_curvature(points):
    points = np.asarray(points, dtype=float)
    arc_length(points)
    curvature = np.zeros(len(points))
    for k in range(1, len(points) - 1):
        a, b = points[k] - points[k - 1], points[k + 1] - points[k]
        denominator = np.linalg.norm(a) * np.linalg.norm(b) * np.linalg.norm(a + b)
        curvature[k] = 0 if denominator < 1e-12 else 2 * (a[0] * b[1] - a[1] * b[0]) / denominator
    if len(points) > 2:
        curvature[0], curvature[-1] = curvature[1], curvature[-2]
    return curvature
