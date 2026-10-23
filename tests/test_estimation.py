import numpy as np
import pytest

from robotlab import estimation as m


def test_gaussian_update_independent_scalar_solution():
    mean, cov, nis, accepted = m.gaussian_update([0.0], [[4.0]], [3.0], [[1.0]], [[5.0]])
    assert mean[0] == pytest.approx(4 / 3)
    assert cov[0, 0] == pytest.approx(20 / 9)
    assert nis == pytest.approx(1)
    assert accepted and np.all(np.linalg.eigvalsh(cov) > 0)
