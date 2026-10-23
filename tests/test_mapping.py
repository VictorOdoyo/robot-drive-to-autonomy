import numpy as np
import pytest

from robotlab import mapping as m


def test_grid_coordinates():
    grid = m.OccupancyGrid(10, 5, 0.2, (-1, -0.5))
    assert grid.cell([-0.9, -0.4]) == (0, 0)
    assert np.allclose(grid.center((0, 0)), [-0.9, -0.4])
    assert not grid.contains(grid.cell([1, -0.4]))
    assert grid.probability()[0, 0] == pytest.approx(0.5)
