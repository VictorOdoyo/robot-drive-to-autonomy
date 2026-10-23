import numpy as np


def path_target(position, path, lookahead=0.4, minimum_progress=0):
    from .geometry import segment_projection
    from .trajectory import arc_length

    path = np.asarray(path, dtype=float)
    distance = arc_length(path)
    if lookahead <= 0 or minimum_progress < 0:
        raise ValueError("invalid lookahead")
    best = (np.inf, minimum_progress)
    for k in range(len(path) - 1):
        _projected, fraction, error = segment_projection(position, path[k], path[k + 1])
        progress = distance[k] + fraction * (distance[k + 1] - distance[k])
        if progress + 1e-9 >= minimum_progress and error < best[0]:
            best = (error, progress)
    progress = max(minimum_progress, best[1])
    target_distance = min(distance[-1], progress + lookahead)
    target = np.array([np.interp(target_distance, distance, path[:, axis]) for axis in range(2)])
    return target, progress, float(best[0])


def pure_pursuit(pose, target, speed=0.6, max_yaw=2):
    from .geometry import rotation

    if speed < 0 or max_yaw <= 0:
        raise ValueError("invalid pursuit limits")
    relative = rotation(pose[2]).T @ (np.asarray(target) - pose[:2])
    distance2 = float(relative @ relative)
    if distance2 < 1e-12:
        return np.zeros(2)
    heading = np.arctan2(relative[1], relative[0])
    forward = speed * max(0, np.cos(heading))
    yaw = 2 * forward * relative[1] / distance2
    if relative[0] <= 0:
        yaw = np.clip(2 * heading, -max_yaw, max_yaw)
    return np.array([forward, np.clip(yaw, -max_yaw, max_yaw)])


def trajectory_control(pose, reference, reference_twist, kx=1.5, ky=3, kheading=2):
    from .geometry import pose_error

    if min(kx, ky, kheading) <= 0:
        raise ValueError("positive trajectory feedback gains required")
    ex, ey, heading = pose_error(pose, reference)
    speed, yaw = reference_twist
    return np.array([speed * np.cos(heading) + kx * ex, yaw + speed * ky * ey + kheading * np.sin(heading)])


def dock_control(pose, target, max_speed=0.25, max_yaw=0.8, position_tolerance=0.02):
    from .geometry import wrap

    delta = np.asarray(target[:2]) - pose[:2]
    rho = np.linalg.norm(delta)
    if min(max_speed, max_yaw, position_tolerance) <= 0:
        raise ValueError("invalid docking limits")
    if rho < position_tolerance:
        return np.array([0.0, np.clip(2 * wrap(target[2] - pose[2]), -max_yaw, max_yaw)])
    alpha = wrap(np.arctan2(delta[1], delta[0]) - pose[2])
    beta = wrap(target[2] - pose[2] - alpha)
    speed = min(max_speed, 0.8 * rho) * max(0, np.cos(alpha))
    return np.array([speed, np.clip(2.5 * alpha - 0.6 * beta, -max_yaw, max_yaw)])


def dynamic_window(current, dt, max_speed=1, max_yaw=2, max_acceleration=0.6, max_yaw_acceleration=2):
    if min(dt, max_speed, max_yaw, max_acceleration, max_yaw_acceleration) <= 0:
        raise ValueError("positive dynamic window constraints required")
    speed, yaw = current
    return np.array(
        [
            [max(0, speed - max_acceleration * dt), min(max_speed, speed + max_acceleration * dt)],
            [max(-max_yaw, yaw - max_yaw_acceleration * dt), min(max_yaw, yaw + max_yaw_acceleration * dt)],
        ]
    )


def dwa(pose, current, target, rectangles, radius=0.22, dt=0.1, horizon=1.0, samples=7):
    from .geometry import integrate, swept_clearance
    from .trajectory import braking_distance

    if samples < 2 or horizon <= 0:
        raise ValueError("invalid rollout sampling")
    window = dynamic_window(current, dt)
    best = None
    for speed in np.linspace(*window[0], samples):
        for yaw in np.linspace(*window[1], samples):
            clearance = swept_clearance(pose, speed, yaw, horizon, radius, rectangles)
            if clearance <= braking_distance(speed, 0.6):
                continue
            end = integrate(pose, speed, yaw, horizon)
            score = (
                np.linalg.norm(end[:2] - target) + 0.15 * abs(yaw) - 0.2 * speed + 0.05 / max(clearance, 0.01)
            )
            if best is None or score < best[0]:
                best = (score, np.array([speed, yaw]))
    return np.zeros(2) if best is None else best[1]


def collision_time(relative_position, relative_velocity, combined_radius, horizon=np.inf):
    position, velocity = np.asarray(relative_position), np.asarray(relative_velocity)
    if combined_radius < 0 or horizon < 0:
        raise ValueError("invalid collision horizon")
    c = position @ position - combined_radius**2
    if c <= 0:
        return 0.0
    a = velocity @ velocity
    b = 2 * position @ velocity
    discriminant = b * b - 4 * a * c
    if a == 0 or discriminant < 0 or b >= 0:
        return np.inf
    collision = (-b - np.sqrt(discriminant)) / (2 * a)
    return float(collision) if collision <= horizon else np.inf


def safe_speed(clearance, deceleration=0.6, latency=0.1, position_sigma=0, sigma_multiplier=3):
    if deceleration <= 0 or min(latency, position_sigma, sigma_multiplier) < 0:
        raise ValueError("invalid stopping constraints")
    available = max(0, clearance - sigma_multiplier * position_sigma)
    return float(
        max(0, np.sqrt((deceleration * latency) ** 2 + 2 * deceleration * available) - deceleration * latency)
    )
