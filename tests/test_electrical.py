import numpy as np
import pytest

from robotlab import electrical as m


def test_motor_stall_and_back_emf():
    motor = m.Motor()
    assert motor.derivative(0, 0, 12)[0] == pytest.approx(12 / motor.inductance)
    assert np.allclose(
        motor.derivative(0, 100, motor.constant * 100), [0, -motor.viscous * 100 / motor.inertia]
    )


def test_motor_power_conservation():
    motor = m.Motor()
    for current, omega, voltage, load in [(5, 100, 12, 0.1), (-3, 200, 5, -0.2)]:
        p = m.motor_power(motor, current, omega, voltage, load)
        assert p["input"] == pytest.approx(sum(p[k] for k in ("copper", "friction", "shaft", "stored_rate")))


def test_winding_time_constant():
    motor = m.Motor()
    current = m.current_step(motor, 0, 0, 10, motor.inductance / motor.resistance)
    assert current == pytest.approx(10 / motor.resistance * (1 - np.exp(-1)))


def test_loaded_motor_equilibrium():
    motor = m.Motor()
    state = m.motor_equilibrium(motor, 12, 0.3)
    assert np.allclose(motor.derivative(*state, 12, 0.3), 0, atol=1e-10)
    assert m.stall_torque(motor, 12) > 0.3


def test_exact_motor_step_composes():
    motor = m.Motor()
    first = m.motor_step(motor, [0, 0], 12, 0.1, 0.01)
    twice = m.motor_step(motor, first, 12, 0.1, 0.01)
    assert np.allclose(twice, m.motor_step(motor, [0, 0], 12, 0.1, 0.02))


def test_bridge_regeneration_balance():
    p = m.bridge(0.5, 24, -4)
    assert p["bus_current"] < 0
    assert 24 * p["bus_current"] == pytest.approx(p["voltage"] * (-4) + p["loss"])


def test_deadtime_current_polarity():
    positive = m.deadtime_voltage(0.3, 24, 2, 1e-6, 20000)
    negative = m.deadtime_voltage(0.3, 24, -2, 1e-6, 20000)
    assert (positive + negative) / 2 == pytest.approx(0.3 * 24)
    assert positive < negative


def test_pwm_ripple_extrema():
    assert m.pwm_ripple(24, 0.5, 0.001, 20000) == pytest.approx(0.3)
    assert m.pwm_ripple(24, 1, 0.001, 20000) == 0
    assert m.switching_loss(24, -10, 20000, 100e-9) > 0
