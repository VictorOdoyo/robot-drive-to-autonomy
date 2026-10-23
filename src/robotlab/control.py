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


def motor_feedforward(current_reference, omega, resistance, constant, current_slew=0, inductance=0):
    if min(resistance, constant) <= 0 or inductance < 0:
        raise ValueError("invalid feedforward model")
    return (
        resistance * np.asarray(current_reference)
        + constant * np.asarray(omega)
        + inductance * np.asarray(current_slew)
    )


def slew(previous, target, rate, dt, deceleration=None):
    if rate <= 0 or dt <= 0 or (deceleration is not None and deceleration <= 0):
        raise ValueError("positive slew bounds required")
    bound = rate if deceleration is None or abs(target) > abs(previous) else deceleration
    return float(previous + np.clip(target - previous, -bound * dt, bound * dt))


def jerk_step(speed, acceleration, target_speed, max_acceleration, max_jerk, dt):
    if min(max_acceleration, max_jerk, dt) <= 0:
        raise ValueError("positive motion bounds required")
    desired = np.clip((target_speed - speed) / dt, -max_acceleration, max_acceleration)
    next_acceleration = slew(acceleration, desired, max_jerk, dt)
    next_speed = speed + 0.5 * (acceleration + next_acceleration) * dt
    return next_speed, next_acceleration
