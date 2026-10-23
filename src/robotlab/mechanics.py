from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chassis:
    mass: float = 18.0
    yaw_inertia: float = 0.65
    wheel_inertia: float = 0.006
    radius: float = 0.08
    track: float = 0.42

    def __post_init__(self):
        if not all(
            np.isfinite(v) and v > 0
            for v in (self.mass, self.yaw_inertia, self.wheel_inertia, self.radius, self.track)
        ):
            raise ValueError("positive finite chassis parameters required")

    def wheel_mass_matrix(self):
        translation = self.mass * self.radius**2 / 4 * np.ones((2, 2))
        rotation = self.yaw_inertia * self.radius**2 / self.track**2 * np.array([[1, -1], [-1, 1]])
        return translation + rotation + self.wheel_inertia * np.eye(2)


def gear_output_torque(motor_torque, motor_speed, ratio=12.0, efficiency=0.9):
    if ratio <= 0 or not 0 < efficiency <= 1:
        raise ValueError("invalid gearbox")
    power = np.asarray(motor_torque) * motor_speed
    factor = np.where(power >= 0, efficiency, 1 / efficiency)
    torque = np.asarray(motor_torque) * ratio * factor
    loss = power - torque * np.asarray(motor_speed) / ratio
    return torque, loss


def reflected_inertia(chassis, motor_inertia, ratio):
    if motor_inertia < 0 or ratio <= 0:
        raise ValueError("invalid reflected inertia")
    return chassis.wheel_mass_matrix() + motor_inertia * ratio**2 * np.eye(2)
