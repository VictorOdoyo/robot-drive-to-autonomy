import numpy as np
import pytest

from robotlab import safety as m


def test_fault_latch_requires_healthy_dwell():
    latch = m.FaultLatch(0.5)
    assert not latch.update(0, "overcurrent")
    assert not latch.update(0.1, reset=True)
    assert not latch.update(0.4, reset=True)
    assert latch.update(0.7, reset=True)
    assert np.isfinite(latch.dwell)
    assert latch.dwell == pytest.approx(0.5)


def test_overload_trip_and_cooling():
    overload = m.Overload(2, 10, 1)
    assert overload.step(10, 0.2)
    before = overload.integral
    overload.step(0, 0.2)
    assert overload.integral < before


def test_watchdog_stops_stale_command():
    watchdog = m.CommandWatchdog(0.2)
    watchdog.receive(1, [0.5, 0.1])
    assert np.allclose(watchdog.output(1.1)[0], [0.5, 0.1])
    command, expired = watchdog.output(1.21)
    assert expired and np.allclose(command, 0)


def test_debounce_resets_after_healthy_sample():
    debounce = m.Debounce(0.1)
    assert not debounce.update(0, True)
    assert not debounce.update(0.09, False)
    assert not debounce.update(0.1, True)
    assert debounce.update(0.21, True)


def test_voltage_hysteresis():
    interlock = m.VoltageInterlock()
    assert not interlock.update(19)
    assert not interlock.update(21)
    assert interlock.update(23)
    assert interlock.update(21)


def test_slip_detection_accounts_for_sensor_noise():
    assert m.slip_detected(0.5, 0.1, 0.001, 0.001)[0]
    assert not m.slip_detected(0.5, 0.1, 0.1, 0.1)[0]


def test_localization_timeout_and_uncertainty():
    assert m.localization_healthy(np.eye(4) * 0.01, 0.5)
    assert not m.localization_healthy(np.eye(4), 0.5)
    assert not m.localization_healthy(np.eye(4) * 0.01, 3)


def test_fault_priority():
    assert m.inhibit_reason(overcurrent=True, stale_command=True) == "overcurrent"
    assert m.inhibit_reason(emergency=True, overcurrent=True) == "emergency"
    assert m.inhibit_reason() is None


def test_rearm_motion_interlock():
    assert m.rearm_allowed(0, 0, 0.5, True, 1)
    assert not m.rearm_allowed(0.1, 0, 0.5, True, 1)
    assert not m.rearm_allowed(0, 0, 0.5, False, 1)


def test_nonfinite_commands_disable_actuation():
    command, valid = m.bounded_command([np.nan, 1], [1, 2])
    assert not valid and np.allclose(command, 0)
    assert np.allclose(m.bounded_command([2, -4], [1, 2])[0], [1, -2])


def test_disturbance_boundaries():
    event = m.Disturbance(0.2, 0.4, "grade", 0.1)
    assert not event.active(0.199)
    assert event.active(0.2) and not event.active(0.4)
    assert m.disturbance_value([event], "grade", 0.4) == 0


def test_fault_journal_records_edges():
    journal = m.FaultJournal()
    journal.update(0, "watchdog")
    journal.update(0.1, "watchdog")
    journal.update(0.2, None)
    assert len(journal.events) == 2
    assert journal.events[-1]["previous"] == "watchdog"
