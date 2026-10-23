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
