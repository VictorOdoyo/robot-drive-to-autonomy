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


def test_thermal_time_constant():
    assert m.thermal_step(25, 10, 25, 2, 100, 200) == pytest.approx(25 + 20 * (1 - np.exp(-1)))
    assert m.thermal_step(100, 0, 25, 2, 100, 200) < 100


def test_thermal_derate():
    assert m.thermal_limit(85, 20) == pytest.approx(10)
    assert m.thermal_limit(105, 20) == 0
    assert m.thermal_limit(25, 20) == 20


def test_dc_link_energy_balance():
    voltage = m.dc_link_step(24, 100, 80, 0.02, 0.01)
    assert 0.5 * 0.02 * (voltage**2 - 24**2) == pytest.approx(0.2)


def test_regen_dispatch_full_battery():
    result = m.regenerative_dispatch(-100, 0, 70)
    assert result == {"battery_power": 0, "brake_power": 70, "unabsorbed_power": 30}
    assert sum([30, 70]) == 100


def test_auxiliary_energy_penalizes_slow_routes():
    slow = m.route_energy(10, 0.25)
    fast = m.route_energy(10, 1)
    assert slow - fast == pytest.approx(12 * (40 - 10))


def test_return_reserve():
    result = m.reserve_margin(1000, 500, 200, 0.2)
    assert result["required_j"] == pytest.approx(875)
    assert result["feasible"]
    assert not m.reserve_margin(800, 500, 200, 0.2)["feasible"]
