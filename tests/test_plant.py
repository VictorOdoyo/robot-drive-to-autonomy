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
