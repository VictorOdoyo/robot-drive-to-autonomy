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
