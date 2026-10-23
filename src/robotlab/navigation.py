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
