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
