import numpy as np
import pytest

from robotlab import kinematics as m


def test_wheel_transform():
    drive = m.DriveGeometry(0.08, 0.081, 0.4)
    assert np.allclose(drive.forward(drive.inverse(1, 0.5)), [1, 0.5])
    assert m.DriveGeometry().forward([-1, 1])[0] == pytest.approx(0)


def test_encoder_circle():
    drive = m.DriveGeometry(track=0.4)
    delta = drive.inverse(0.5, 1.0) * 2 * np.pi
    assert np.allclose(m.odometry([0, 0, 0], delta, drive), 0, atol=1e-14)


def test_correlated_encoders():
    common = np.ones((2, 2))
    covariance = m.twist_covariance(common)
    assert covariance[1, 1] == pytest.approx(0, abs=1e-14)
    assert covariance[0, 0] > 0


def test_curvature_preserving_limit():
    drive = m.DriveGeometry()
    wheels, scale = m.limit_twist(2, 3, 10, drive)
    assert max(abs(wheels)) == pytest.approx(10)
    assert np.allclose(drive.forward(wheels), np.array([2, 3]) * scale)
