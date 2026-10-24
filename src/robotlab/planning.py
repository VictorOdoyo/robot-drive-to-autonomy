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
    """Exact cell-boundary supercover, including cells touched only at a corner."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    delta = end - start
    crossing = [0.0, 1.0]
    for axis in range(2):
        if abs(delta[axis]) < 1e-15:
            continue
        low, high = sorted([start[axis] + 0.5, end[axis] + 0.5])
        for boundary in range(int(np.ceil(low)), int(np.floor(high)) + 1):
            fraction = (boundary - 0.5 - start[axis]) / delta[axis]
            if 0 <= fraction <= 1:
                crossing.append(fraction)
    crossing = np.unique(crossing)
    samples = np.r_[crossing, (crossing[:-1] + crossing[1:]) / 2]
    for fraction in samples:
        point = start + fraction * delta
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


def space_time_path(blocked, start, goal, reservations, horizon=100):
    from collections import deque

    start, goal = tuple(start), tuple(goal)
    if blocked[start] or blocked[goal] or start in reservations.get(0, set()):
        raise ValueError("invalid space-time endpoints")
    queue = deque([(start, 0)])
    parent = {(start, 0): None}
    while queue:
        cell, tick = queue.popleft()
        if cell == goal:
            result, key = [], (cell, tick)
            while key is not None:
                result.append(key)
                key = parent[key]
            return result[::-1]
        if tick >= horizon:
            continue
        for nxt, _ in [(cell, 0), *neighbors(cell, blocked, False)]:
            key = (nxt, tick + 1)
            if nxt in reservations.get(tick + 1, set()) or key in parent:
                continue
            parent[key] = (cell, tick)
            queue.append(key)
    raise ValueError("no path within reservation horizon")


def visit_order(distance_matrix, start=0, return_to_start=True):
    distance_matrix = np.asarray(distance_matrix)
    n = len(distance_matrix)
    if distance_matrix.shape != (n, n) or n > 16 or np.any(distance_matrix < 0):
        raise ValueError("nonnegative square distances for <=16 stops required")
    if not 0 <= start < n:
        raise ValueError("invalid start index")
    states = {(1 << start, start): (0.0, (start,))}
    for mask in range(1 << n):
        for end in range(n):
            if (mask, end) not in states:
                continue
            cost, path = states[mask, end]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                key = (mask | (1 << nxt), nxt)
                candidate = cost + distance_matrix[end, nxt]
                if candidate < states.get(key, (np.inf, ()))[0]:
                    states[key] = (candidate, (*path, nxt))
    candidates = []
    for (mask, end), (cost, path) in states.items():
        if mask == (1 << n) - 1:
            candidates.append(
                (
                    cost + (distance_matrix[end, start] if return_to_start else 0),
                    (*path, start) if return_to_start else path,
                )
            )
    if not candidates:
        raise ValueError("no finite tour")
    return min(candidates)


def rank_frontiers(frontiers, distances, information_gain, travel_penalty=1):
    frontiers = np.asarray(frontiers, dtype=int)
    gains = np.asarray(information_gain)
    if len(frontiers) != len(gains) or travel_penalty <= 0 or np.any(gains < 0):
        raise ValueError("invalid frontier utilities")
    scores = np.array(
        [
            gain / (travel_penalty + distances[tuple(cell)])
            for cell, gain in zip(frontiers, gains, strict=True)
        ]
    )
    return np.argsort(-scores), scores


def coverage_path(blocked, start, row_stride=2):
    if row_stride < 1:
        raise ValueError("positive coverage stride required")
    current = tuple(start)
    result = [current]
    reachable = np.isfinite(distance_map(blocked, current))
    for row in range(0, blocked.shape[0], row_stride):
        columns = list(np.flatnonzero(reachable[row]))
        if (row // row_stride) % 2:
            columns.reverse()
        for col in columns:
            target = (row, int(col))
            if target == current:
                continue
            connection, _ = astar(blocked, current, target)
            result.extend(connection[1:])
            current = target
    return result


def heading_path(blocked, start, goal, turn_cost=0.4):
    import heapq

    if turn_cost <= 0:
        raise ValueError("positive turning cost required")
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = tuple(start)
    queue = [(0.0, start)]
    best, parent = {start: 0.0}, {}
    while queue:
        cost, state = heapq.heappop(queue)
        if cost > best[state]:
            continue
        row, col, heading = state
        if (row, col) == tuple(goal):
            path = [state]
            while state != start:
                state = parent[state]
                path.append(state)
            return path[::-1], cost
        dr, dc = directions[heading]
        candidates = [((row, col, (heading + 1) % 4), turn_cost), ((row, col, (heading - 1) % 4), turn_cost)]
        nr, nc = row + dr, col + dc
        if 0 <= nr < blocked.shape[0] and 0 <= nc < blocked.shape[1] and not blocked[nr, nc]:
            candidates.append(((nr, nc, heading), 1.0))
        for nxt, edge in candidates:
            if cost + edge < best.get(nxt, np.inf):
                best[nxt], parent[nxt] = cost + edge, state
                heapq.heappush(queue, (cost + edge, nxt))
    raise ValueError("unreachable heading path")


def energy_path(graph, start, goal, budget=np.inf):
    import heapq

    queue = [(0.0, start)]
    best, parent = {start: 0.0}, {}
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > best[node]:
            continue
        if node == goal:
            path = [node]
            while node != start:
                node = parent[node]
                path.append(node)
            return path[::-1], cost
        for nxt, energy in graph.get(node, []):
            if energy < 0 or not np.isfinite(energy):
                raise ValueError("energy edges must be finite and nonnegative")
            candidate = cost + energy
            if candidate <= budget and candidate < best.get(nxt, np.inf):
                best[nxt], parent[nxt] = candidate, node
                heapq.heappush(queue, (candidate, nxt))
    raise ValueError("no route within energy budget")


def elastic_smooth(path, blocked, weight=0.3, iterations=100):
    if not 0 < weight < 0.5 or iterations < 0:
        raise ValueError("invalid smoothing parameters")
    points = np.asarray(path, dtype=float).copy()
    for _ in range(iterations):
        previous = points.copy()
        for k in range(1, len(points) - 1):
            candidate = points[k] + weight * (points[k - 1] + points[k + 1] - 2 * points[k])
            if line_free(blocked, points[k - 1], candidate) and line_free(blocked, candidate, points[k + 1]):
                points[k] = candidate
        if np.max(np.abs(points - previous), initial=0) < 1e-7:
            break
    return points


def route_valid(path, blocked):
    from itertools import pairwise

    return bool(
        len(path) > 0
        and all(line_free(blocked, a, b) for a, b in pairwise(path))
        and line_free(blocked, path[-1], path[-1])
    )
