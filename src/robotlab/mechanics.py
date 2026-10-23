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


def friction_torque(rates, coulomb=0.03, viscous=0.004, smoothing=0.05):
    if min(coulomb, viscous) < 0 or smoothing <= 0:
        raise ValueError("invalid friction parameters")
    rates = np.asarray(rates)
    return coulomb * np.tanh(rates / smoothing) + viscous * rates


def road_load(speed, mass, radius, grade=0, rolling_coefficient=0.015, gravity=9.80665):
    if min(mass, radius, gravity) <= 0 or rolling_coefficient < 0:
        raise ValueError("invalid road load parameters")
    force = mass * gravity * (np.sin(grade) + rolling_coefficient * np.cos(grade) * np.tanh(speed / 0.02))
    return np.ones(2) * force * radius / 2


def traction_limit(normal_force, friction, lateral_force=0):
    normal_force, friction, lateral_force = np.broadcast_arrays(normal_force, friction, lateral_force)
    if np.any(normal_force < 0) or np.any(friction < 0):
        raise ValueError("negative contact parameter")
    return np.sqrt(np.maximum((normal_force * friction) ** 2 - lateral_force**2, 0))


def normal_loads(mass, track, height, lateral_acceleration, gravity=9.80665):
    if min(mass, track, gravity) <= 0 or height < 0:
        raise ValueError("invalid support geometry")
    transfer = mass * lateral_acceleration * height / track
    loads = mass * gravity / 2 + np.array([-transfer, transfer])
    return np.maximum(loads, 0), bool(np.any(loads <= 0))


def tire_force(surface_speed, ground_speed, normal_force, friction=0.6, stiffness=120):
    if normal_force < 0 or friction < 0 or stiffness <= 0:
        raise ValueError("invalid tire model")
    difference = surface_speed - ground_speed
    bound = friction * normal_force
    return float(np.clip(stiffness * difference, -bound, bound))


def slip_ratio(surface_speed, ground_speed, regularization=0.1):
    if regularization <= 0:
        raise ValueError("regularization must be positive")
    return (surface_speed - ground_speed) / max(abs(surface_speed), abs(ground_speed), regularization)


def backlash_torque(
    motor_angle, wheel_angle, motor_rate, wheel_rate, ratio, half_gap=0.005, stiffness=50, damping=0.05
):
    if ratio <= 0 or min(half_gap, stiffness, damping) < 0:
        raise ValueError("invalid backlash parameters")
    error = motor_angle / ratio - wheel_angle
    if abs(error) <= half_gap:
        return 0.0
    elastic = stiffness * (error - np.sign(error) * half_gap)
    viscous = damping * (motor_rate / ratio - wheel_rate)
    # A unilateral tooth contact cannot pull the opposite tooth flank.
    return float(np.sign(error) * max(0, np.sign(error) * (elastic + viscous)))


def wheel_acceleration(chassis, rates, drive_torque, external_torque=(0, 0), motor_inertia=0, ratio=1):
    matrix = reflected_inertia(chassis, motor_inertia, ratio)
    net = np.asarray(drive_torque) - external_torque - friction_torque(rates)
    return np.linalg.solve(matrix, net)


def kinetic_energy(chassis, rates, motor_inertia=0, ratio=1):
    rates = np.asarray(rates)
    return float(0.5 * rates @ reflected_inertia(chassis, motor_inertia, ratio) @ rates)
