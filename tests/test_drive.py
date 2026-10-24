import numpy as np
import pytest

from robotlab import drive as m


def test_independent_drive_controllers():
    controller = m.DriveController()
    controller.current_loops[0].integral = 1
    assert controller.current_loops[1].integral == 0
    assert controller.current_limit == pytest.approx(8)
    assert np.allclose(controller.reference, 0)


def test_twist_slew_and_curvature():
    controller = m.DriveController()
    assert np.allclose(m.set_twist(controller, [1, 0], 0.01), [0.12, 0.12])
    for _ in range(200):
        m.set_twist(controller, [1, 0], 0.01)
    assert np.allclose(controller.reference, [12.5, 12.5])
