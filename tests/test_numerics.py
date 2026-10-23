import numpy as np
import pytest

from robotlab import numerics as m


def test_jacobian_against_analytic():
    def fun(x):
        return np.array([x[0] ** 2 + x[1], np.sin(x[1])])

    assert np.allclose(m.jacobian(fun, [2, 0.3]), [[4, 1], [0, np.cos(0.3)]], atol=1e-8)
    assert m.jacobian(lambda x: x**3, [2])[0, 0] == pytest.approx(12)
