import numpy as np
import pytest

from robotlab import planning as m


def test_adjacency_blocks_corner_cutting():
    blocked = np.array([[False, True], [True, False]])
    assert list(m.neighbors((0, 0), blocked)) == []
    assert m.octile((0, 0), (3, 2)) == pytest.approx(1 + 2 * np.sqrt(2))


def test_astar_known_optimum_and_detour():
    blocked = np.zeros((8, 8), bool)
    path, cost = m.astar(blocked, (0, 0), (4, 3))
    assert cost == pytest.approx(1 + 3 * np.sqrt(2))
    blocked[1:7, 3] = True
    path, cost = m.astar(blocked, (2, 1), (2, 5))
    assert cost > 4 and all(not blocked[p] for p in path)


def test_disconnected_region_unreachable():
    blocked = np.zeros((5, 5), bool)
    blocked[:, 2] = True
    distance = m.distance_map(blocked, (1, 1))
    assert distance[3, 1] == 2
    assert np.isinf(distance[1, 3])
