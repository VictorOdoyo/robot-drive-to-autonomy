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


def test_hot_motor_current_is_reduced():
    controller = m.DriveController()
    controller.reference[:] = 10
    reference = m.velocity_tick(controller, [0, 0], [25, 100], 0.01)
    assert reference[0] > 0 and reference[1] == 0


def test_current_duty_is_voltage_limited():
    controller = m.DriveController()
    controller.current_reference[:] = 8
    duty = m.current_tick(controller, [0, 0], [100, -100], 24, 0.002)
    assert np.allclose(duty, [1, -1])
    assert np.all(np.abs(duty) <= 1)


def test_cascade_tracks_speed_without_excess_current():
    _, states = m.regulated_drive(1.5, 0.5)
    assert states[-1, 4] == pytest.approx(0.5, abs=0.04)
    assert np.max(abs(states[:, :2])) < 8.5
    assert abs(states[-1, 5]) < 1e-10


def test_back_emf_observer_recovers_wheel_speed():
    from robotlab.electrical import Motor

    motor = Motor()
    rates = np.array([5.0, -3.0])
    current = np.array([2.0, -1.0])
    derivative = np.array([30.0, -40.0])
    voltage = motor.resistance * current + motor.inductance * derivative + motor.constant * 12 * rates
    assert np.allclose(m.back_emf_wheel_speed(voltage, current, derivative, motor), rates)
