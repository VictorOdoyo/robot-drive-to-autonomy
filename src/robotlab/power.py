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


def battery_step(pack, soc, polarization, current, dt):
    if dt < 0:
        raise ValueError("negative timestep")
    next_soc = soc - current * dt / (3600 * pack.capacity_ah)
    decay = np.exp(-dt / pack.polarization_time)
    next_polarization = polarization * decay + pack.polarization_resistance * current * (1 - decay)
    if not 0 <= next_soc <= 1:
        raise ValueError("charge-state boundary exceeded")
    return float(next_soc), float(next_polarization)


def terminal_voltage(pack, soc, polarization, current):
    return pack.ocv(soc) - polarization - pack.resistance * current


def chemical_energy(pack, soc):
    if not 0 <= soc <= 1:
        raise ValueError("SOC outside [0,1]")
    return (
        3600
        * pack.capacity_ah
        * (pack.empty_voltage * soc + 0.5 * (pack.full_voltage - pack.empty_voltage) * soc**2)
    )


def current_for_power(pack, soc, polarization, power):
    source = pack.ocv(soc) - polarization
    if source <= 0:
        raise ValueError("nonpositive battery source voltage")
    discriminant = source**2 - 4 * pack.resistance * power
    if discriminant < 0:
        raise ValueError("requested power exceeds battery maximum")
    # Rationalized expression avoids cancellation near zero power.
    return float(2 * power / (source + np.sqrt(discriminant)))
