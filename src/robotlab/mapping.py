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
