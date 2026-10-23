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


def speed_profile(points, max_speed=1, max_acceleration=0.6, max_lateral=0.8, start_speed=0, end_speed=0):
    if min(max_speed, max_acceleration, max_lateral) <= 0 or min(start_speed, end_speed) < 0:
        raise ValueError("invalid retiming limits")
    distance = arc_length(points)
    curvature = np.abs(path_curvature(points))
    speed = np.minimum(max_speed, np.sqrt(max_lateral / np.maximum(curvature, 1e-12)))
    speed[0] = min(speed[0], start_speed)
    for k in range(1, len(speed)):
        speed[k] = min(
            speed[k], np.sqrt(speed[k - 1] ** 2 + 2 * max_acceleration * (distance[k] - distance[k - 1]))
        )
    speed[-1] = min(speed[-1], end_speed)
    for k in range(len(speed) - 2, -1, -1):
        speed[k] = min(
            speed[k], np.sqrt(speed[k + 1] ** 2 + 2 * max_acceleration * (distance[k + 1] - distance[k]))
        )
    return speed


def profile_time(points, speed):
    distance = arc_length(points)
    speed = np.asarray(speed)
    if speed.shape != distance.shape or np.any(speed < 0):
        raise ValueError("one nonnegative speed per waypoint required")
    denominator = speed[:-1] + speed[1:]
    if np.any((denominator <= 0) & (np.diff(distance) > 0)):
        raise ValueError("positive traversal speed required on each segment")
    intervals = np.divide(
        2 * np.diff(distance), denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return np.r_[0, np.cumsum(intervals)]


def circle(time, radius=1, speed=0.5):
    if radius <= 0:
        raise ValueError("positive circle radius required")
    time = np.asarray(time)
    angle = speed * time / radius
    pose = np.column_stack([radius * np.sin(angle), radius * (1 - np.cos(angle)), angle])
    twist = np.tile([speed, speed / radius], (len(time), 1))
    return pose, twist


def figure_eight(time, scale=1, frequency=0.15):
    if scale <= 0 or frequency <= 0:
        raise ValueError("positive figure-eight dimensions required")
    time = np.asarray(time)
    angle = frequency * time
    x, y = scale * np.sin(angle), scale * np.sin(2 * angle) / 2
    dx, dy = scale * frequency * np.cos(angle), scale * frequency * np.cos(2 * angle)
    ddx, ddy = -scale * frequency**2 * np.sin(angle), -2 * scale * frequency**2 * np.sin(2 * angle)
    speed = np.hypot(dx, dy)
    yaw = np.unwrap(np.arctan2(dy, dx))
    yaw_rate = (dx * ddy - dy * ddx) / np.maximum(speed**2, 1e-12)
    return np.column_stack([x, y, yaw]), np.column_stack([speed, yaw_rate])
