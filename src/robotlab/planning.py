import numpy as np


def neighbors(cell, blocked, diagonal=True):
    row, col = cell
    height, width = blocked.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)] + (
        [(-1, -1), (-1, 1), (1, -1), (1, 1)] if diagonal else []
    ):
        nr, nc = row + dr, col + dc
        if not (0 <= nr < height and 0 <= nc < width) or blocked[nr, nc]:
            continue
        if dr and dc and (blocked[row, nc] or blocked[nr, col]):
            continue
        yield (nr, nc), float(np.hypot(dr, dc))


def octile(a, b):
    dy, dx = np.abs(np.asarray(a) - b)
    return float(max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy))
