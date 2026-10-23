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
