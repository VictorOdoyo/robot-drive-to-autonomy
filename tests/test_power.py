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


def test_battery_power_solution():
    pack = m.Battery()
    for power in (0, 100, -80):
        current = m.current_for_power(pack, 0.6, 0.2, power)
        assert current * m.terminal_voltage(pack, 0.6, 0.2, current) == pytest.approx(power)
    with pytest.raises(ValueError):
        m.current_for_power(pack, 0.6, 0, 1e6)


def test_bms_blocks_full_pack_regeneration():
    pack = m.Battery()
    low, high = m.battery_current_bounds(pack, 1, 0, 0.1)
    assert low == 0
    assert high > 0
    assert m.battery_current_bounds(pack, 0, 0, 0.1)[1] == 0
