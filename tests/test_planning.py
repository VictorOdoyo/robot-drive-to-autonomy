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


def test_line_of_sight_checks_diagonal_corner():
    blocked = np.zeros((3, 3), bool)
    blocked[0, 1] = True
    assert not m.line_free(blocked, (0, 0), (2, 2))
    assert m.line_free(blocked, (1, 0), (2, 0))


def test_shortcut_open_route():
    blocked = np.zeros((10, 10), bool)
    path, _ = m.astar(blocked, (1, 1), (8, 7))
    result = m.shortcut(path, blocked)
    assert result == [(1, 1), (8, 7)]


def test_risk_cost_avoids_expensive_corridor():
    blocked = np.zeros((7, 9), bool)
    risk = np.ones((7, 9))
    risk[3, 2:7] = 20
    path, cost = m.astar(blocked, (3, 0), (3, 8), risk)
    assert all(not (p[0] == 3 and 2 <= p[1] <= 6) for p in path)
    assert cost < 20
    assert m.clearance_cost([0, 1])[0] > m.clearance_cost([0, 1])[1]


def test_time_expanded_path_waits_for_obstacle():
    blocked = np.zeros((1, 3), bool)
    path = m.space_time_path(blocked, (0, 0), (0, 2), {1: {(0, 1)}})
    assert path[1] == ((0, 0), 1)
    assert path[-1] == ((0, 2), 3)


def test_multi_stop_optimum_against_enumeration():
    import itertools

    distances = np.array([[0, 3, 8, 2], [3, 0, 4, 5], [8, 4, 0, 1], [2, 5, 1, 0]])
    cost, path = m.visit_order(distances)
    brute = min(
        sum(distances[a, b] for a, b in zip((0, *p), (*p, 0), strict=True))
        for p in itertools.permutations([1, 2, 3])
    )
    assert cost == brute and path[0] == path[-1] == 0


def test_frontier_utility_balances_gain_and_distance():
    distance = np.array([[0.0, 1, 10]])
    order, scores = m.rank_frontiers([[0, 1], [0, 2]], distance, [3, 5])
    assert order[0] == 0 and scores[0] > scores[1]


def test_coverage_connects_disjoint_row_intervals():
    blocked = np.zeros((5, 7), bool)
    blocked[1:4, 3] = True
    path = m.coverage_path(blocked, (0, 0), 1)
    assert len(set(path)) == np.sum(~blocked)
    assert all(not blocked[p] for p in path)
