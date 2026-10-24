from dataclasses import dataclass, field

import numpy as np

from .electrical import Motor
from .mechanics import Chassis
from .power import Battery


@dataclass(frozen=True)
class PlantParameters:
    motor: Motor = field(default_factory=Motor)
    chassis: Chassis = field(default_factory=Chassis)
    battery: Battery = field(default_factory=Battery)
    ratio: float = 12.0
    auxiliary_power: float = 12.0
    ambient: float = 25.0
    rolling_coefficient: float = 0.015
    tire_stiffness: float = 120.0
    bridge_resistance: float = 0.02

    def __post_init__(self):
        if (
            min(self.ratio, self.tire_stiffness) <= 0
            or min(self.auxiliary_power, self.rolling_coefficient, self.bridge_resistance) < 0
        ):
            raise ValueError("invalid coupled plant parameters")


def initial_state(parameters=PlantParameters(), pose=(0, 0, 0), soc=0.8):
    parameters.battery.ocv(soc)
    state = np.zeros(18)
    state[6:9] = pose
    state[9] = soc
    state[11:14] = parameters.ambient
    return state
