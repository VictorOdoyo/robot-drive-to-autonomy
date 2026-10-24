import numpy as np
import pytest

from robotlab import plant as m


def test_plant_initial_state():
    state = m.initial_state(pose=(1, 2, 0.3), soc=0.7)
    assert np.allclose(state[6:9], [1, 2, 0.3])
    assert state[9] == pytest.approx(0.7)
    assert np.allclose(state[11:14], 25)


def test_bus_kirchhoff_and_power_balance():
    parameters = m.PlantParameters()
    state = m.initial_state(parameters)
    state[:2] = [4, -2]
    duty = np.array([0.4, 0.2])
    voltage, current = m.bus_operating_point(parameters, state, duty)
    assert voltage == pytest.approx(parameters.battery.ocv(0.8) - parameters.battery.resistance * current)
    assert voltage * current == pytest.approx(voltage * (duty @ state[:2]) + 12)


def test_traction_saturates_on_low_friction_floor():
    parameters = m.PlantParameters()
    state = m.initial_state(parameters)
    state[2:4] = 100
    forces, slip, lift = m.contact_forces(parameters, state, (0.05, 0.6))
    assert forces[0] == pytest.approx(0.05 * parameters.chassis.mass * 9.80665 / 2)
    assert forces[1] > forces[0]
    assert np.all(forces * slip >= 0) and not lift


def test_coupled_acceleration_symmetry():
    parameters = m.PlantParameters()
    state = m.initial_state(parameters)
    state[:2] = 2
    derivative, losses, observation = m.mechanical_electrical_rhs(parameters, state, [0.3, 0.3])
    assert derivative[2] == pytest.approx(derivative[3])
    assert derivative[5] == 0
    assert all(v >= 0 for v in losses.values())
    assert observation["bus_voltage"] > 20


def test_component_thermal_and_charge_directions():
    parameters = m.PlantParameters()
    derivative, loss = m.battery_thermal_rhs(parameters, m.initial_state(parameters), 10, [20, 10])
    assert derivative[0] < 0 and derivative[1] > 0
    assert derivative[2] > derivative[3] > 0
    assert loss == pytest.approx(9)


def test_stored_energy_includes_wheel_rotor_inertia():
    parameters = m.PlantParameters()
    state = m.initial_state(parameters)
    initial = m.stored_energy(parameters, state)
    state[2] = 10
    expected = 0.5 * (parameters.chassis.wheel_inertia + parameters.motor.inertia * 12**2) * 100
    assert m.stored_energy(parameters, state) - initial == pytest.approx(expected, abs=1e-9)


def test_stationary_auxiliary_energy_ledger():
    parameters = m.PlantParameters()
    state = m.initial_state(parameters)
    initial = m.stored_energy(parameters, state)
    for _ in range(50):
        state = m.step(parameters, state, [0, 0], 0.002)
    assert initial - m.stored_energy(parameters, state) == pytest.approx(state[16] + state[17], abs=1e-7)
    assert np.allclose(state[:9], 0)


def test_driven_energy_audit_under_grade_and_slip():
    parameters = m.PlantParameters()
    initial = m.initial_state(parameters)
    state = initial.copy()
    for _ in range(250):
        state = m.step(parameters, state, [0.3, 0.35], 0.001, grade=0.03, traction=(0.08, 0.6))
    assert state[4] > 0 and state[5] > 0
    assert abs(m.energy_residual(parameters, initial, state)) < 0.002


def test_physical_telemetry():
    parameters = m.PlantParameters()
    state = m.initial_state(parameters)
    data = m.telemetry(parameters, state, [0, 0])
    assert np.allclose(data["slip_speed"], 0)
    assert data["battery_current"] > 0
    assert data["stored_j"] > 0
