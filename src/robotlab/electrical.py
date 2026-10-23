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
