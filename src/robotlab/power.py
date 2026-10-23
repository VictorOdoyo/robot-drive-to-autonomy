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


def battery_current_bounds(
    pack, soc, polarization, dt, min_voltage=20, max_voltage=30, discharge=30, charge=10
):
    if dt <= 0 or min(discharge, charge) < 0 or max_voltage <= min_voltage:
        raise ValueError("invalid BMS constraints")
    source = pack.ocv(soc) - polarization
    lower = max(-charge, (source - max_voltage) / pack.resistance, -(1 - soc) * 3600 * pack.capacity_ah / dt)
    upper = min(discharge, (source - min_voltage) / pack.resistance, soc * 3600 * pack.capacity_ah / dt)
    if lower > upper:
        raise ValueError("no feasible battery current")
    return lower, upper


def thermal_step(temperature, loss, ambient, thermal_resistance, thermal_capacity, dt):
    if min(thermal_resistance, thermal_capacity) <= 0 or dt < 0 or loss < 0:
        raise ValueError("invalid thermal parameters")
    equilibrium = ambient + loss * thermal_resistance
    return float(
        equilibrium + (temperature - equilibrium) * np.exp(-dt / (thermal_resistance * thermal_capacity))
    )


def thermal_limit(temperature, full_current, derate_start=70, shutdown=100):
    if shutdown <= derate_start or full_current < 0:
        raise ValueError("invalid derating thresholds")
    return float(full_current * np.clip((shutdown - temperature) / (shutdown - derate_start), 0, 1))


def dc_link_step(voltage, source_power, load_power, capacitance, dt):
    if min(voltage, capacitance) <= 0 or dt < 0:
        raise ValueError("invalid DC link parameters")
    energy = 0.5 * capacitance * voltage**2 + (source_power - load_power) * dt
    if energy <= 0:
        raise ValueError("DC link energy depleted")
    return float(np.sqrt(2 * energy / capacitance))


def regenerative_dispatch(requested_power, charge_power_limit, chopper_limit):
    if min(charge_power_limit, chopper_limit) < 0:
        raise ValueError("negative absorption capacity")
    regenerative = max(0, -requested_power)
    battery = min(regenerative, charge_power_limit)
    brake = min(regenerative - battery, chopper_limit)
    return {
        "battery_power": max(requested_power, 0) - battery,
        "brake_power": brake,
        "unabsorbed_power": regenerative - battery - brake,
    }


def route_energy(length, speed, mass=18, rolling=0.015, efficiency=0.75, auxiliary=12, grade=0):
    if length < 0 or speed <= 0 or mass <= 0 or not 0 < efficiency <= 1 or min(rolling, auxiliary) < 0:
        raise ValueError("invalid route energy parameters")
    work = mass * 9.80665 * (rolling * np.cos(grade) + np.sin(grade)) * length
    traction = work / efficiency if work >= 0 else work * efficiency
    return traction + auxiliary * length / speed


def reserve_margin(available_energy, mission_energy, return_energy, reserve_fraction=0.15):
    if min(available_energy, mission_energy, return_energy) < 0 or not 0 <= reserve_fraction < 1:
        raise ValueError("invalid energy budget")
    required = (mission_energy + return_energy) / (1 - reserve_fraction)
    return {
        "required_j": required,
        "margin_j": available_energy - required,
        "feasible": available_energy >= required,
    }
