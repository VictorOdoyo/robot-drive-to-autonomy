import numpy as np
import pytest

from robotlab import geometry as m


def test_rotations():
    assert m.wrap(3 * np.pi) == pytest.approx(-np.pi)
    assert np.allclose(m.rotation(0.7).T @ m.rotation(0.7), np.eye(2))


def test_frame_inverse():
    pose = np.array([2.0, -1.0, 0.8])
    assert np.allclose(m.compose(pose, m.inverse(pose)), 0, atol=1e-14)
    assert np.allclose(m.compose([0, 0, np.pi / 2], [1, 0, 0]), [0, 1, np.pi / 2])
