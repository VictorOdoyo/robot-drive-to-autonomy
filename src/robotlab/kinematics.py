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
