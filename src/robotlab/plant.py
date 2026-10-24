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


def bus_operating_point(parameters, state, duty):
    battery = parameters.battery
    source = battery.ocv(state[9]) - state[10]
    drive_current = float(np.asarray(duty) @ state[:2])
    effective = source - battery.resistance * drive_current
    discriminant = effective**2 - 4 * battery.resistance * parameters.auxiliary_power
    if effective <= 0 or discriminant <= 0:
        raise ValueError("DC bus voltage collapse")
    voltage = (effective + np.sqrt(discriminant)) / 2
    current = drive_current + parameters.auxiliary_power / voltage
    return float(voltage), float(current)


def contact_forces(parameters, state, traction=(0.6, 0.6)):
    from .mechanics import normal_loads, tire_force

    chassis = parameters.chassis
    traction = np.asarray(traction, dtype=float)
    if traction.shape != (2,) or np.any(traction < 0):
        raise ValueError("two nonnegative tire friction coefficients required")
    ground = state[4] + np.array([-1, 1]) * state[5] * chassis.track / 2
    loads, lift = normal_loads(chassis.mass, chassis.track, 0.18, state[4] * state[5])
    forces = np.array(
        [
            tire_force(
                chassis.radius * state[2 + k], ground[k], loads[k], traction[k], parameters.tire_stiffness
            )
            for k in range(2)
        ]
    )
    slip = chassis.radius * state[2:4] - ground
    return forces, slip, lift
