import numpy as np


class OccupancyGrid:
    def __init__(self, width, height, resolution=0.1, origin=(0, 0)):
        if min(width, height) <= 0 or resolution <= 0:
            raise ValueError("positive grid dimensions required")
        self.resolution = float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.log_odds = np.zeros((int(height), int(width)))
        self.observed = np.zeros_like(self.log_odds, dtype=bool)

    def cell(self, point):
        col, row = np.floor((np.asarray(point) - self.origin) / self.resolution).astype(int)
        return int(row), int(col)

    def center(self, cell):
        row, col = cell
        return self.origin + self.resolution * (np.array([col, row]) + 0.5)

    def contains(self, cell):
        row, col = cell
        return 0 <= row < self.log_odds.shape[0] and 0 <= col < self.log_odds.shape[1]

    def probability(self):
        from scipy.special import expit

        return expit(self.log_odds)


def rasterize(grid, rectangles):
    half = grid.resolution / 2
    for row in range(grid.log_odds.shape[0]):
        for col in range(grid.log_odds.shape[1]):
            x, y = grid.center((row, col))
            occupied = any(
                x + half >= r[0] and x - half <= r[2] and y + half >= r[1] and y - half <= r[3]
                for r in rectangles
            )
            grid.log_odds[row, col] = 8.0 if occupied else -8.0
            grid.observed[row, col] = True
    return grid


def ray_cells(start, end):
    y0, x0 = start
    y1, x1 = end
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx + dy
    cells = []
    while True:
        cells.append((y0, x0))
        if (y0, x0) == (y1, x1):
            return cells
        twice = 2 * error
        if twice >= dy:
            error += dy
            x0 += sx
        if twice <= dx:
            error += dx
            y0 += sy


def insert_scan(grid, pose, angles, ranges, max_range, free_increment=-0.4, occupied_increment=0.85):
    if len(angles) != len(ranges) or max_range <= 0:
        raise ValueError("invalid scan")
    start = grid.cell(pose[:2])
    for angle, distance in zip(angles, ranges, strict=True):
        if not np.isfinite(distance) or distance < 0:
            continue
        hit = distance < max_range - 1e-9
        distance = min(distance, max_range)
        endpoint = np.asarray(pose[:2]) + distance * np.array(
            [np.cos(pose[2] + angle), np.sin(pose[2] + angle)]
        )
        cells = ray_cells(start, grid.cell(endpoint))
        for index, cell in enumerate(cells):
            if not grid.contains(cell):
                continue
            increment = occupied_increment if hit and index == len(cells) - 1 else free_increment
            grid.log_odds[cell] = np.clip(grid.log_odds[cell] + increment, -8, 8)
            grid.observed[cell] = True


def distance_field(grid, threshold=0.65, unknown_occupied=True):
    from scipy.ndimage import distance_transform_edt

    occupied = grid.probability() >= threshold
    if unknown_occupied:
        occupied |= ~grid.observed
    free = np.pad(~occupied, 1, constant_values=False)
    distance = distance_transform_edt(free)[1:-1, 1:-1] * grid.resolution
    return np.maximum(distance - grid.resolution * np.sqrt(2) / 2, 0)


def inflated_mask(grid, radius, margin=0, unknown_occupied=True):
    if min(radius, margin) < 0:
        raise ValueError("negative footprint")
    return distance_field(grid, unknown_occupied=unknown_occupied) <= radius + margin


def frontiers(grid, free_threshold=0.35):
    from scipy.ndimage import binary_dilation

    free = grid.observed & (grid.probability() < free_threshold)
    adjacent_unknown = binary_dilation(~grid.observed, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    return np.argwhere(free & adjacent_unknown)


def map_entropy(grid):
    probability = np.clip(grid.probability(), 1e-15, 1 - 1e-15)
    return float(np.sum(-probability * np.log(probability) - (1 - probability) * np.log1p(-probability)))
