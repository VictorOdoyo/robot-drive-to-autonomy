import numpy as np
import pytest

from robotlab import power as m


def test_battery_ocv():
    pack = m.Battery()
    assert pack.ocv(0.5) == pytest.approx((pack.empty_voltage + pack.full_voltage) / 2)
    assert np.isfinite(pack.ocv(1))
    with pytest.raises(ValueError):
        pack.ocv(1.1)


def test_battery_coulomb_count_and_relaxation():
    pack = m.Battery()
    soc, polar = m.battery_step(pack, 1, 0, 12, 360)
    assert soc == pytest.approx(0.9)
    assert polar == pytest.approx(12 * pack.polarization_resistance, rel=1e-6)
    assert m.battery_step(pack, soc, polar, 0, 20)[1] == pytest.approx(polar / np.e)


def test_chemical_energy_gradient():
    pack = m.Battery()
    delta = 1e-5
    gradient = (m.chemical_energy(pack, 0.6 + delta) - m.chemical_energy(pack, 0.6 - delta)) / (2 * delta)
    assert gradient == pytest.approx(3600 * pack.capacity_ah * pack.ocv(0.6), rel=1e-9)
