import numpy as np
import pytest

from robotlab import kinematics as m


def test_wheel_transform():
    drive = m.DriveGeometry(0.08, 0.081, 0.4)
    assert np.allclose(drive.forward(drive.inverse(1, 0.5)), [1, 0.5])
    assert m.DriveGeometry().forward([-1, 1])[0] == pytest.approx(0)
