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


def mechanical_electrical_rhs(parameters, state, duty, grade=0, traction=(0.6, 0.6), external_force=0):
    from .electrical import copper_resistance
    from .mechanics import friction_torque

    chassis, motor = parameters.chassis, parameters.motor
    duty = np.clip(np.asarray(duty, dtype=float), -1, 1)
    bus, battery_current = bus_operating_point(parameters, state, duty)
    current, wheels = state[:2], state[2:4]
    resistance = np.array([copper_resistance(motor.resistance, t) for t in state[11:13]])
    voltage = duty * bus - parameters.bridge_resistance * current
    forces, slip, lift = contact_forces(parameters, state, traction)
    derivative = np.zeros(9)
    derivative[:2] = (
        voltage - resistance * current - motor.constant * parameters.ratio * wheels
    ) / motor.inductance
    bearing = friction_torque(wheels) + motor.viscous * parameters.ratio**2 * wheels
    equivalent_inertia = chassis.wheel_inertia + motor.inertia * parameters.ratio**2
    derivative[2:4] = (
        motor.constant * parameters.ratio * current - bearing - chassis.radius * forces
    ) / equivalent_inertia
    rolling = chassis.mass * 9.80665 * parameters.rolling_coefficient * np.tanh(state[4] / 0.03)
    aerodynamic = 0.1 * state[4] * abs(state[4])
    grade_force = chassis.mass * 9.80665 * np.sin(grade)
    derivative[4] = (np.sum(forces) - rolling - aerodynamic - grade_force - external_force) / chassis.mass
    derivative[5] = ((forces[1] - forces[0]) * chassis.track / 2 - 0.03 * state[5]) / chassis.yaw_inertia
    derivative[6:9] = [state[4] * np.cos(state[8]), state[4] * np.sin(state[8]), state[5]]
    losses = {
        "copper": float(resistance @ current**2),
        "bridge": float(parameters.bridge_resistance * (current @ current)),
        "bearing": float(bearing @ wheels),
        "slip": float(forces @ slip),
        "road": float((rolling + aerodynamic) * state[4] + 0.03 * state[5] ** 2),
    }
    return (
        derivative,
        losses,
        {
            "bus_voltage": bus,
            "battery_current": battery_current,
            "external_power": (grade_force + external_force) * state[4],
            "wheel_lift": lift,
        },
    )
