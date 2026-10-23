import numpy as np
import pytest

from robotlab import electrical as m


def test_motor_stall_and_back_emf():
    motor = m.Motor()
    assert motor.derivative(0, 0, 12)[0] == pytest.approx(12 / motor.inductance)
    assert np.allclose(
        motor.derivative(0, 100, motor.constant * 100), [0, -motor.viscous * 100 / motor.inertia]
    )
