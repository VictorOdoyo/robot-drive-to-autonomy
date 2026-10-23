import numpy as np
import pytest

from robotlab import power as m


def test_battery_ocv():
    pack = m.Battery()
    assert pack.ocv(0.5) == pytest.approx((pack.empty_voltage + pack.full_voltage) / 2)
    assert np.isfinite(pack.ocv(1))
    with pytest.raises(ValueError):
        pack.ocv(1.1)
