from dataclasses import dataclass

import numpy as np


@dataclass
class PI:
    kp: float
    ki: float
    tracking_gain: float = 20.0
    integral: float = 0.0

    def __post_init__(self):
        if min(self.kp, self.ki, self.tracking_gain) < 0:
            raise ValueError("negative controller gains")

    def step(self, error, dt, lower, upper, feedforward=0):
        if dt <= 0 or lower > upper or not np.isfinite(error + feedforward):
            raise ValueError("invalid controller input")
        raw = self.kp * error + self.integral + feedforward
        output = float(np.clip(raw, lower, upper))
        self.integral += dt * (self.ki * error + self.tracking_gain * (output - raw))
        return output


def reset_pi(controller, output, error=0, feedforward=0):
    if not np.isfinite(output + error + feedforward):
        raise ValueError("nonfinite reset")
    controller.integral = output - controller.kp * error - feedforward


def current_gains(resistance, inductance, bandwidth):
    if min(resistance, inductance, bandwidth) <= 0:
        raise ValueError("positive winding parameters required")
    return inductance * bandwidth, resistance * bandwidth


def velocity_gains(inertia, damping, torque_constant, bandwidth, damping_ratio=1):
    if min(inertia, torque_constant, bandwidth, damping_ratio) <= 0 or damping < 0:
        raise ValueError("invalid speed loop parameters")
    kp = (2 * damping_ratio * bandwidth * inertia - damping) / torque_constant
    ki = inertia * bandwidth**2 / torque_constant
    if kp < 0:
        raise ValueError("requested bandwidth below passive damping")
    return kp, ki
