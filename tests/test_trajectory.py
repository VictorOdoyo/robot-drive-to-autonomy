import numpy as np
import pytest

from robotlab import trajectory as m


def test_quintic_boundary_conditions():
    coefficients = m.quintic([1, 0.2, 0.1], [4, -0.3, 0.2], 2)
    assert np.allclose(m.polynomial_state(coefficients, 0), [1, 0.2, 0.1])
    assert m.polynomial_state(coefficients, 2) == pytest.approx([4, -0.3, 0.2])


def test_trapezoid_and_triangle():
    for distance in (0.1, 10):
        time, position, speed = m.trapezoid(distance, 1, 2)
        assert position[-1] == pytest.approx(distance)
        assert speed[0] == speed[-1] == 0
        assert np.max(speed) <= 1 + 1e-12
        assert np.max(abs(np.diff(speed) / np.diff(time))) <= 2 + 1e-10


def test_resample_duplicate_waypoints():
    points = m.resample_path([[0, 0], [0, 0], [1, 0], [1, 1]], 0.1)
    assert np.allclose(points[[0, -1]], [[0, 0], [1, 1]])
    assert m.arc_length(points)[-1] == pytest.approx(2)
