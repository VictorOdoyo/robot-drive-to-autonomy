from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Motor:
    resistance: float = 0.35
    inductance: float = 0.0015
    constant: float = 0.065
    inertia: float = 0.00012
    viscous: float = 0.0001

    def __post_init__(self):
        values = (self.resistance, self.inductance, self.constant, self.inertia)
        if not all(np.isfinite(v) and v > 0 for v in values) or self.viscous < 0:
            raise ValueError("invalid motor parameters")

    def derivative(self, current, omega, voltage, load_torque=0):
        return np.array(
            [
                (voltage - self.resistance * current - self.constant * omega) / self.inductance,
                (self.constant * current - self.viscous * omega - load_torque) / self.inertia,
            ]
        )


def motor_power(motor, current, omega, voltage, load_torque=0):
    derivative = motor.derivative(current, omega, voltage, load_torque)
    stored_rate = motor.inductance * current * derivative[0] + motor.inertia * omega * derivative[1]
    return {
        "input": voltage * current,
        "copper": motor.resistance * current**2,
        "friction": motor.viscous * omega**2,
        "shaft": load_torque * omega,
        "stored_rate": stored_rate,
    }


def motor_energy(motor, current, omega):
    return 0.5 * (motor.inductance * current**2 + motor.inertia * omega**2)


def current_step(motor, current, omega, voltage, dt):
    if dt < 0:
        raise ValueError("negative timestep")
    equilibrium = (voltage - motor.constant * omega) / motor.resistance
    return equilibrium + (current - equilibrium) * np.exp(-dt * motor.resistance / motor.inductance)


def motor_equilibrium(motor, voltage, load_torque=0):
    omega = (motor.constant * voltage - motor.resistance * load_torque) / (
        motor.constant**2 + motor.resistance * motor.viscous
    )
    current = (voltage - motor.constant * omega) / motor.resistance
    return np.array([current, omega])


def stall_torque(motor, voltage):
    return motor.constant * voltage / motor.resistance


def motor_state_space(motor):
    matrix = np.array(
        [
            [-motor.resistance / motor.inductance, -motor.constant / motor.inductance],
            [motor.constant / motor.inertia, -motor.viscous / motor.inertia],
        ]
    )
    inputs = np.array([[1 / motor.inductance, 0], [0, -1 / motor.inertia]])
    return matrix, inputs


def motor_step(motor, state, voltage, load_torque, dt):
    from .numerics import discretize

    ad, bd = discretize(*motor_state_space(motor), dt)
    return ad @ state + bd @ [voltage, load_torque]


def bridge(duty, bus_voltage, current, on_resistance=0.02):
    if bus_voltage <= 0 or on_resistance < 0:
        raise ValueError("invalid bridge parameters")
    duty = float(np.clip(duty, -1, 1))
    loss = on_resistance * current**2
    voltage = duty * bus_voltage - on_resistance * current
    return {"voltage": voltage, "bus_current": duty * current, "loss": loss, "duty": duty}


def deadtime_voltage(duty, bus_voltage, current, deadtime, switching_frequency):
    if deadtime < 0 or switching_frequency <= 0 or deadtime * switching_frequency >= 0.5:
        raise ValueError("invalid PWM timing")
    effective = np.clip(duty, -1, 1) - 2 * deadtime * switching_frequency * np.sign(current)
    return float(np.clip(effective, -1, 1) * bus_voltage)


def pwm_ripple(bus_voltage, duty, inductance, frequency):
    if min(bus_voltage, inductance, frequency) <= 0 or not 0 <= duty <= 1:
        raise ValueError("invalid unipolar PWM parameters")
    return bus_voltage * duty * (1 - duty) / (inductance * frequency)


def switching_loss(bus_voltage, current, frequency, rise_fall_time, devices=2):
    if min(bus_voltage, frequency, rise_fall_time, devices) < 0:
        raise ValueError("negative switching parameter")
    return 0.5 * devices * bus_voltage * abs(current) * frequency * rise_fall_time


def copper_resistance(reference, temperature, reference_temperature=25, coefficient=0.00393):
    resistance = reference * (1 + coefficient * (temperature - reference_temperature))
    if reference <= 0 or resistance <= 0:
        raise ValueError("nonphysical winding resistance")
    return resistance


def identify_winding(current, derivative, voltage, omega, constant):
    current, derivative, voltage, omega = map(np.asarray, (current, derivative, voltage, omega))
    design = np.column_stack([current, derivative])
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("insufficient winding excitation")
    resistance, inductance = np.linalg.lstsq(design, voltage - constant * omega, rcond=None)[0]
    if min(resistance, inductance) <= 0:
        raise ValueError("nonphysical winding fit")
    return float(resistance), float(inductance)


def torque_limit(motor, omega, bus_voltage, current_limit):
    if bus_voltage <= 0 or current_limit <= 0:
        raise ValueError("positive voltage and current limit required")
    lower = np.maximum(-current_limit, (-bus_voltage - motor.constant * np.asarray(omega)) / motor.resistance)
    upper = np.minimum(current_limit, (bus_voltage - motor.constant * np.asarray(omega)) / motor.resistance)
    return motor.constant * lower, motor.constant * upper


def dynamic_brake(motor, omega, brake_resistance):
    if brake_resistance < 0:
        raise ValueError("negative brake resistance")
    current = -motor.constant * omega / (motor.resistance + brake_resistance)
    return {
        "current": current,
        "torque": motor.constant * current,
        "resistor_power": current**2 * brake_resistance,
        "copper_power": current**2 * motor.resistance,
    }
