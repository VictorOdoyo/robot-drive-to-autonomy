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


def astar(blocked, start, goal, cost=None):
    import heapq

    blocked = np.asarray(blocked, dtype=bool)
    start, goal = tuple(start), tuple(goal)
    height, width = blocked.shape
    for cell in (start, goal):
        if not (0 <= cell[0] < height and 0 <= cell[1] < width) or blocked[cell]:
            raise ValueError("start and goal must be free map cells")
    cost = np.ones(blocked.shape) if cost is None else np.asarray(cost)
    if cost.shape != blocked.shape or np.any(~np.isfinite(cost)) or np.any(cost < 1):
        raise ValueError("finite traversal costs >=1 required")
    queue = [(octile(start, goal), 0.0, start)]
    best, parent = {start: 0.0}, {}
    while queue:
        _, distance, cell = heapq.heappop(queue)
        if distance > best[cell] + 1e-12:
            continue
        if cell == goal:
            path = [cell]
            while cell != start:
                cell = parent[cell]
                path.append(cell)
            return path[::-1], distance
        for nxt, length in neighbors(cell, blocked):
            candidate = distance + length * (cost[cell] + cost[nxt]) / 2
            if candidate < best.get(nxt, np.inf) - 1e-12:
                best[nxt], parent[nxt] = candidate, cell
                heapq.heappush(queue, (candidate + octile(nxt, goal), candidate, nxt))
    raise ValueError("goal is unreachable")


def distance_map(blocked, start):
    import heapq

    blocked = np.asarray(blocked, dtype=bool)
    if not (0 <= start[0] < blocked.shape[0] and 0 <= start[1] < blocked.shape[1]) or blocked[start]:
        raise ValueError("free start required")
    distances = np.full(blocked.shape, np.inf)
    distances[start] = 0
    queue = [(0.0, tuple(start))]
    while queue:
        distance, cell = heapq.heappop(queue)
        if distance > distances[cell]:
            continue
        for nxt, length in neighbors(cell, blocked):
            candidate = distance + length
            if candidate < distances[nxt]:
                distances[nxt] = candidate
                heapq.heappush(queue, (candidate, nxt))
    return distances


def line_free(blocked, start, end):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    count = max(1, int(np.ceil(np.linalg.norm(end - start) * 4)))
    for point in np.linspace(start, end, count + 1):
        # Integer coordinates denote cell centers. Check both sides of exact boundaries.
        for epsilon in ((-1e-10, -1e-10), (-1e-10, 1e-10), (1e-10, -1e-10), (1e-10, 1e-10)):
            cell = np.floor(point + 0.5 + epsilon).astype(int)
            if (
                np.any(cell < 0)
                or cell[0] >= blocked.shape[0]
                or cell[1] >= blocked.shape[1]
                or blocked[tuple(cell)]
            ):
                return False
    return True


def shortcut(path, blocked):
    if len(path) < 2:
        return list(path)
    result, index = [path[0]], 0
    while index < len(path) - 1:
        target = len(path) - 1
        while target > index + 1 and not line_free(blocked, path[index], path[target]):
            target -= 1
        if not line_free(blocked, path[index], path[target]):
            raise ValueError("input path contains unsafe segment")
        result.append(path[target])
        index = target
    return result


def clearance_cost(clearance, weight=1, scale=0.3):
    if weight < 0 or scale <= 0 or np.any(np.asarray(clearance) < 0):
        raise ValueError("invalid clearance cost")
    return 1 + weight * np.exp(-np.asarray(clearance) / scale)
