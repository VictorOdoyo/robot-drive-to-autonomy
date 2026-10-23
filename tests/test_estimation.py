import numpy as np
import pytest

from robotlab import estimation as m


def test_gaussian_update_independent_scalar_solution():
    mean, cov, nis, accepted = m.gaussian_update([0.0], [[4.0]], [3.0], [[1.0]], [[5.0]])
    assert mean[0] == pytest.approx(4 / 3)
    assert cov[0, 0] == pytest.approx(20 / 9)
    assert nis == pytest.approx(1)
    assert accepted and np.all(np.linalg.eigvalsh(cov) > 0)


def test_bias_corrected_arc_and_jacobian():
    state = np.array([0.0, 0.0, 0.0, 0.1])
    result = m.motion(state, 1, 1.1, np.pi)
    assert np.allclose(result[:3], [0, 2, np.pi], atol=1e-14)
    transition, _ = m.motion_jacobians(state, 1, 1.1, 0.1)
    assert transition[2, 3] == pytest.approx(-0.1)
