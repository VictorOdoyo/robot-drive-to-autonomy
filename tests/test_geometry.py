import numpy as np
import pytest

from robotlab import geometry as m


def test_rotations():
    assert m.wrap(3 * np.pi) == pytest.approx(-np.pi)
    assert np.allclose(m.rotation(0.7).T @ m.rotation(0.7), np.eye(2))
