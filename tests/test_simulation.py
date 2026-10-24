import numpy as np
import pytest

from robotlab import simulation as m


def test_multirate_clock_and_event_boundaries():
    from robotlab.cases import Scenario
    from robotlab.safety import Disturbance

    case = Scenario(duration=0.2, events=(Disturbance(0.033, 0.171, "grade", 0.1),))
    time = m.simulation_times(case)
    assert np.min(np.diff(time)) > 1e-8
    assert 0.033 in time and 0.171 in time
    clock = m.PeriodicClock(0.05)
    assert sum(clock.due(t) for t in time) == 5
    assert time[-1] == pytest.approx(0.2)


def test_seeded_sensor_stream_and_dropout():
    from robotlab.cases import Scenario
    from robotlab.plant import initial_state

    first, second = m.SensorSuite(Scenario()), m.SensorSuite(Scenario())
    state = initial_state()
    assert first.gyro(state) == second.gyro(state)
    first.sample_wheels(state, 0)
    state[14:16] = 0.1
    rates = first.sample_wheels(state, 0.01)
    assert np.all(rates > 0)
    first.sample_wheels(state, 0.02, True)
    assert first.age_stamp == pytest.approx(0.01)
