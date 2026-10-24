import numpy as np
import pytest

from robotlab import drive as m


def test_independent_drive_controllers():
    controller = m.DriveController()
    controller.current_loops[0].integral = 1
    assert controller.current_loops[1].integral == 0
    assert controller.current_limit == pytest.approx(8)
    assert np.allclose(controller.reference, 0)
