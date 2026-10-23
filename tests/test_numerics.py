import numpy as np
import pytest

from robotlab import numerics as m


def test_jacobian_against_analytic():
    def fun(x):
        return np.array([x[0] ** 2 + x[1], np.sin(x[1])])

    assert np.allclose(m.jacobian(fun, [2, 0.3]), [[4, 1], [0, np.cos(0.3)]], atol=1e-8)
    assert m.jacobian(lambda x: x**3, [2])[0, 0] == pytest.approx(12)


def test_rk4_order():
    def error(dt):
        x = np.array([1.0])
        for i in range(round(1 / dt)):
            x = m.rk4(lambda t, y: -y, i * dt, x, dt)
        return abs(x[0] - np.exp(-1))

    assert error(0.1) / error(0.05) > 15


def test_zoh_integrator_and_decay():
    ad, bd = m.discretize(np.array([[-2.0]]), np.array([[3.0]]), 0.4)
    assert ad[0, 0] == pytest.approx(np.exp(-0.8))
    assert bd[0, 0] == pytest.approx(1.5 * (1 - np.exp(-0.8)))
    assert m.discretize(np.zeros((1, 1)), np.ones((1, 1)), 0.3)[1][0, 0] == pytest.approx(0.3)


def test_integrated_white_acceleration():
    q = m.process_noise(np.array([[0.0, 1], [0, 0]]), np.diag([0.0, 2.0]), 0.5)
    assert np.allclose(q, [[2 * 0.5**3 / 3, 0.5**2], [0.5**2, 1.0]])


def test_invalid_covariances():
    assert np.allclose(m.covariance([[1, 1], [1, 1]]), np.ones((2, 2)))
    for bad in ([[1, 2], [0, 1]], [[1, 2], [2, 1]], [[float("nan")]]):
        with pytest.raises(ValueError):
            m.covariance(bad)
