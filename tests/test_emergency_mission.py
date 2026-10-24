import numpy as np

from robotlab.cases import get_case
from robotlab.simulation import run


def test_emergency_stop_is_latched_and_brakes_the_robot():
    result = run(get_case("emergency-stop"))
    time, state = result["time"], result["state"]
    assert np.max(state[time < 2, 4]) > 0.3
    assert np.max(abs(state[time > 4, 4])) < 0.03
    assert np.allclose(result["command"][time >= 2.05], 0)
    assert result["journal"][-1]["reason"] == "emergency"
    assert result["clearance"].min() > 0
    assert not result["completed"]
