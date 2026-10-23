from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DriveGeometry:
    left_radius: float = 0.08
    right_radius: float = 0.08
    track: float = 0.42

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in (self.left_radius, self.right_radius, self.track)):
            raise ValueError("geometry must be finite and positive")

    def forward(self, wheels):
        left, right = np.asarray(wheels) * [self.left_radius, self.right_radius]
        return np.array([(left + right) / 2, (right - left) / self.track])

    def inverse(self, speed, yaw_rate):
        return np.array(
            [
                (speed - yaw_rate * self.track / 2) / self.left_radius,
                (speed + yaw_rate * self.track / 2) / self.right_radius,
            ]
        )


def odometry(pose, delta_angles, geometry=DriveGeometry()):
    from .geometry import integrate

    distance, angle = geometry.forward(delta_angles)
    return integrate(pose, distance, angle, 1.0)


def wheel_jacobian(geometry=DriveGeometry()):
    return np.array(
        [
            [geometry.left_radius / 2, geometry.right_radius / 2],
            [-geometry.left_radius / geometry.track, geometry.right_radius / geometry.track],
        ]
    )


def twist_covariance(wheel_covariance, geometry=DriveGeometry()):
    covariance = np.asarray(wheel_covariance)
    if covariance.shape != (2, 2) or not np.allclose(covariance, covariance.T):
        raise ValueError("wheel covariance must be symmetric 2x2")
    if np.linalg.eigvalsh(covariance).min() < 0:
        raise ValueError("wheel covariance must be positive semidefinite")
    jacobian = wheel_jacobian(geometry)
    return jacobian @ covariance @ jacobian.T


def limit_twist(speed, yaw_rate, max_wheel_rate, geometry=DriveGeometry()):
    if max_wheel_rate <= 0:
        raise ValueError("wheel limit must be positive")
    wheels = geometry.inverse(speed, yaw_rate)
    scale = min(1.0, max_wheel_rate / max(np.max(np.abs(wheels)), 1e-15))
    return wheels * scale, scale


def calibrate_radii(wheel_increments, body_increments, track):
    wheels, body = np.asarray(wheel_increments), np.asarray(body_increments)
    if wheels.ndim != 2 or wheels.shape[1] != 2 or body.shape != wheels.shape or track <= 0:
        raise ValueError("need paired Nx2 wheel and body measurements")
    matrix = np.zeros((2 * len(wheels), 2))
    matrix[0::2] = wheels / 2
    matrix[1::2] = wheels * np.array([-1, 1]) / track
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("wheel radii are unobservable")
    radii = np.linalg.lstsq(matrix, body.reshape(-1), rcond=None)[0]
    return DriveGeometry(*radii, track)


def calibrate_track(wheel_increments, yaw_increments, left_radius, right_radius):
    wheels, yaw = np.asarray(wheel_increments), np.asarray(yaw_increments)
    difference = wheels[:, 1] * right_radius - wheels[:, 0] * left_radius
    if yaw @ yaw <= 1e-12:
        raise ValueError("rotation excitation is required")
    track = float(yaw @ difference / (yaw @ yaw))
    if track <= 0:
        raise ValueError("inconsistent rotation or wheel sign")
    return track


def slipping_twist(wheels, slip, geometry=DriveGeometry()):
    slip = np.asarray(slip)
    if slip.shape != (2,) or np.any((slip < 0) | (slip > 1)):
        raise ValueError("traction loss must be in [0,1] for each wheel")
    return geometry.forward(np.asarray(wheels) * (1 - slip))


def wheel_forces(force, moment, geometry=DriveGeometry()):
    return np.array([force / 2 - moment / geometry.track, force / 2 + moment / geometry.track])


def body_wrench(forces, geometry=DriveGeometry()):
    left, right = forces
    return np.array([left + right, (right - left) * geometry.track / 2])
