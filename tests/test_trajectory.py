import numpy as np
import pytest

from robotlab import trajectory as m


def test_quintic_boundary_conditions():
    coefficients = m.quintic([1, 0.2, 0.1], [4, -0.3, 0.2], 2)
    assert np.allclose(m.polynomial_state(coefficients, 0), [1, 0.2, 0.1])
    assert m.polynomial_state(coefficients, 2) == pytest.approx([4, -0.3, 0.2])
