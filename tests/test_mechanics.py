import numpy as np
import pytest

from robotlab import mechanics as m


def test_chassis_effective_inertia():
    robot = m.Chassis()
    inertia = robot.wheel_mass_matrix()
    assert np.all(np.linalg.eigvalsh(inertia) > 0)
    assert np.ones(2) @ inertia @ np.ones(2) == pytest.approx(
        robot.mass * robot.radius**2 + 2 * robot.wheel_inertia
    )


def test_gearbox_motoring_and_backdrive():
    for torque in (1, -1):
        output, loss = m.gear_output_torque(torque, 120, 12, 0.8)
        assert loss >= 0
        assert torque * 120 == pytest.approx(output * 10 + loss)


def test_reflected_rotational_energy():
    chassis = m.Chassis()
    rates = np.array([2.0, -3.0])
    matrix = m.reflected_inertia(chassis, 0.00012, 12)
    difference = 0.5 * rates @ (matrix - chassis.wheel_mass_matrix()) @ rates
    assert difference == pytest.approx(0.5 * 0.00012 * np.sum((12 * rates) ** 2))


def test_friction_is_dissipative():
    rates = np.linspace(-20, 20, 101)
    assert np.all(m.friction_torque(rates) * rates >= 0)
    assert m.friction_torque(0) == 0
