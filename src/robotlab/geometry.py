import numpy as np


def wrap(angle):
    """Principal angle in [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def compose(a, b):
    xy = np.asarray(a[:2]) + rotation(a[2]) @ b[:2]
    return np.r_[xy, wrap(a[2] + b[2])]


def inverse(pose):
    return np.r_[-rotation(pose[2]).T @ pose[:2], wrap(-pose[2])]


def transform_points(pose, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N,2)")
    return points @ rotation(pose[2]).T + pose[:2]


def integrate(pose, speed, yaw_rate, dt):
    if dt < 0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and nonnegative")
    angle = yaw_rate * dt
    distance = speed * dt * np.sinc(angle / (2 * np.pi))
    displacement = [distance * np.cos(angle / 2), distance * np.sin(angle / 2), angle]
    return compose(pose, displacement)


def pose_error(actual, reference):
    """Reference relative to actual, in actual body axes."""
    return compose(inverse(actual), reference)


def interpolate(a, b, fraction):
    if not 0 <= fraction <= 1:
        raise ValueError("fraction outside [0,1]")
    xy = (1 - fraction) * np.asarray(a[:2]) + fraction * np.asarray(b[:2])
    return np.r_[xy, wrap(a[2] + fraction * wrap(b[2] - a[2]))]


def segment_projection(point, start, end):
    delta = np.asarray(end) - start
    norm2 = float(delta @ delta)
    fraction = 0.0 if norm2 == 0 else float(np.clip((np.asarray(point) - start) @ delta / norm2, 0, 1))
    projected = np.asarray(start) + fraction * delta
    return projected, fraction, float(np.linalg.norm(np.asarray(point) - projected))
