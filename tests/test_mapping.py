import numpy as np
import pytest

from robotlab import mapping as m


def test_grid_coordinates():
    grid = m.OccupancyGrid(10, 5, 0.2, (-1, -0.5))
    assert grid.cell([-0.9, -0.4]) == (0, 0)
    assert np.allclose(grid.center((0, 0)), [-0.9, -0.4])
    assert not grid.contains(grid.cell([1, -0.4]))
    assert grid.probability()[0, 0] == pytest.approx(0.5)


def test_thin_obstacle_rasterization():
    grid = m.rasterize(m.OccupancyGrid(10, 10, 0.1), [[0.495, 0.1, 0.505, 0.9]])
    assert np.all(grid.probability()[2:8, 4:6] > 0.99)


def test_ray_all_octants():
    for end in [(5, 2), (-5, 2), (2, 5), (0, -5)]:
        cells = m.ray_cells((0, 0), end)
        assert cells[0] == (0, 0) and cells[-1] == end
        assert max(np.max(np.abs(np.diff(cells, axis=0)), axis=1)) <= 1
