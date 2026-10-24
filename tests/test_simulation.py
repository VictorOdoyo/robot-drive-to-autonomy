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
