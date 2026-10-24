import numpy as np

from robotlab.cases import get_case
from robotlab.simulation import run


def test_traction_loss_produces_physical_slip_and_recovers():
    result = run(get_case("traction-loss"))
    time = result["time"]
    during = (time > 1.2) & (time < 2.4)
    after = time > 4
    assert np.max(abs(result["slip_speed"][during, 0])) > 0.04
    assert np.mean(abs(result["slip_speed"][during, 0])) > np.mean(abs(result["slip_speed"][after, 0])) * 2
    assert result["completed"]
    assert result["clearance"].min() > 0
