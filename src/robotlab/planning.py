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
