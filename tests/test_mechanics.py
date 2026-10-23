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


def test_grade_energy():
    torque = m.road_load(1, 20, 0.1, grade=np.arcsin(0.1), rolling_coefficient=0)
    assert np.sum(torque) / 0.1 == pytest.approx(20 * 9.80665 * 0.1)


def test_friction_circle():
    assert m.traction_limit(100, 0.5, 30) == pytest.approx(40)
    assert m.traction_limit(100, 0.5, 60) == 0


def test_lateral_load_transfer():
    loads, lift = m.normal_loads(20, 0.4, 0.2, 2)
    assert loads.sum() == pytest.approx(20 * 9.80665)
    assert loads[1] - loads[0] == pytest.approx(40)
    assert not lift
    assert m.normal_loads(20, 0.4, 0.2, 20)[1]


def test_tire_slip_dissipation():
    for surface, ground in [(1, 0), (-1, 0), (0, 2)]:
        force = m.tire_force(surface, ground, 100, 0.4)
        assert abs(force) <= 40
        assert force * (surface - ground) >= 0
    assert m.slip_ratio(0, 0) == 0


def test_backlash_deadzone():
    assert m.backlash_torque(0.001, 0, 0, 0, 1) == 0
    assert m.backlash_torque(0.02, 0, 0, 0, 1) == pytest.approx(0.75)
    assert m.backlash_torque(-0.02, 0, 0, 0, 1) == pytest.approx(-0.75)


def test_chassis_dynamic_power_balance():
    chassis = m.Chassis()
    rates, torque = np.array([3.0, 5.0]), np.array([1.0, 2.0])
    acceleration = m.wheel_acceleration(chassis, rates, torque, motor_inertia=0.0001, ratio=12)
    stored_rate = rates @ m.reflected_inertia(chassis, 0.0001, 12) @ acceleration
    assert stored_rate == pytest.approx(rates @ (torque - m.friction_torque(rates)))
