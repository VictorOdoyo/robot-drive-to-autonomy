import numpy as np
import pytest

from robotlab import navigation as m


def test_lookahead_uses_arc_length():
    target, progress, error = m.path_target([0.8, 0.1], [[0, 0], [1, 0], [1, 1]], 0.5)
    assert np.allclose(target, [1, 0.3])
    assert progress == pytest.approx(0.8)
    assert error == pytest.approx(0.1)


def test_pursuit_steering_direction():
    assert m.pure_pursuit([0, 0, 0], [1, 1])[1] > 0
    assert np.allclose(m.pure_pursuit([0, 0, 0], [1, 0]), [0.6, 0])
    assert m.pure_pursuit([0, 0, 0], [-1, 0])[0] == 0
