from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Battery:
    capacity_ah: float = 12.0
    empty_voltage: float = 21.0
    full_voltage: float = 29.4
    resistance: float = 0.09
    polarization_resistance: float = 0.025
    polarization_time: float = 20.0

    def __post_init__(self):
        if not all(
            np.isfinite(v) and v > 0
            for v in (
                self.capacity_ah,
                self.empty_voltage,
                self.full_voltage,
                self.resistance,
                self.polarization_resistance,
                self.polarization_time,
            )
        ):
            raise ValueError("positive finite battery parameters required")
        if self.full_voltage <= self.empty_voltage:
            raise ValueError("invalid voltage range")

    def ocv(self, soc):
        if not 0 <= soc <= 1:
            raise ValueError("SOC outside [0,1]")
        return self.empty_voltage + (self.full_voltage - self.empty_voltage) * soc
