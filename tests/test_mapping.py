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


def test_scan_hit_and_no_return():
    grid = m.OccupancyGrid(30, 10, 0.1)
    m.insert_scan(grid, [0.05, 0.45, 0], [0], [1.0], 2)
    assert grid.log_odds[4, 10] > 0
    assert grid.log_odds[4, 5] < 0
    m.insert_scan(grid, [0.05, 0.55, 0], [0], [2.0], 2)
    assert grid.log_odds[5, 20] < 0


def test_distance_field_wall_and_boundary():
    grid = m.rasterize(m.OccupancyGrid(20, 20, 0.1), [[0.9, 0, 1.1, 2]])
    distance = m.distance_field(grid)
    assert distance[10, 10] == 0
    assert distance[10, 5] > 0.2
    assert distance[0, 5] < 0.1


def test_narrow_passage_disappears_after_inflation():
    grid = m.rasterize(m.OccupancyGrid(30, 30, 0.1), [[0, 0, 1.3, 3], [1.7, 0, 3, 3]])
    assert not m.inflated_mask(grid, 0.01)[15, 15]
    assert m.inflated_mask(grid, 0.25)[15, 15]


def test_frontier_cells_are_observed_free():
    grid = m.OccupancyGrid(5, 5, 1)
    grid.observed[2, 1:4] = True
    grid.log_odds[2, 1:4] = -4
    result = m.frontiers(grid)
    assert len(result) == 3
    assert np.all(result[:, 0] == 2)


def test_mapping_reduces_entropy():
    grid = m.OccupancyGrid(10, 10, 0.1)
    initial = m.map_entropy(grid)
    m.insert_scan(grid, [0.05, 0.45, 0], [0], [0.7], 1)
    assert m.map_entropy(grid) < initial
    assert initial == pytest.approx(100 * np.log(2))
