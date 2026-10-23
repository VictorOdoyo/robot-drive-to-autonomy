import numpy as np
import pytest

from robotlab import control as m


def test_pi_saturation():
    controller = m.PI(2, 5)
    outputs = [controller.step(10, 0.001, -1, 1) for _ in range(1000)]
    assert np.max(outputs) == 1
    assert controller.integral == pytest.approx(-16.5, abs=0.01)
