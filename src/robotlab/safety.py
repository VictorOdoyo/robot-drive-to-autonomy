from dataclasses import dataclass

import numpy as np


@dataclass
class FaultLatch:
    dwell: float = 0.5
    fault: str | None = None
    healthy_since: float | None = None

    def update(self, time, reason=None, reset=False):
        if self.dwell < 0 or not np.isfinite(time):
            raise ValueError("invalid fault timing")
        if reason is not None:
            self.fault = str(reason)
            self.healthy_since = None
        elif self.fault is not None:
            if self.healthy_since is None:
                self.healthy_since = time
            if reset and time - self.healthy_since >= self.dwell:
                self.fault = None
                self.healthy_since = None
        return self.fault is None


@dataclass
class Overload:
    continuous: float = 8.0
    budget: float = 400.0
    cooling_time: float = 20.0
    integral: float = 0.0

    def step(self, current, dt):
        if min(self.continuous, self.budget, self.cooling_time, dt) <= 0:
            raise ValueError("invalid overload settings")
        excess = max(0, current**2 - self.continuous**2)
        decay = np.exp(-dt / self.cooling_time)
        self.integral = self.integral * decay + excess * self.cooling_time * (1 - decay)
        return self.integral >= self.budget


class CommandWatchdog:
    def __init__(self, timeout=0.25):
        if timeout <= 0:
            raise ValueError("positive watchdog timeout required")
        self.timeout = timeout
        self.stamp = -np.inf
        self.command = np.zeros(2)

    def receive(self, stamp, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (2,) or not np.all(np.isfinite(command)) or stamp < self.stamp:
            raise ValueError("finite ordered chassis command required")
        self.stamp, self.command = stamp, command.copy()

    def output(self, time):
        expired = time - self.stamp > self.timeout or time < self.stamp
        return (np.zeros(2) if expired else self.command.copy()), expired


class Debounce:
    def __init__(self, dwell):
        if dwell < 0:
            raise ValueError("negative debounce dwell")
        self.dwell = dwell
        self.since = None

    def update(self, time, condition):
        if not condition:
            self.since = None
            return False
        if self.since is None:
            self.since = time
        return time - self.since >= self.dwell


@dataclass
class VoltageInterlock:
    trip: float = 20.0
    recover: float = 22.0
    enabled: bool = True

    def update(self, voltage):
        if self.recover <= self.trip or not np.isfinite(voltage):
            raise ValueError("invalid voltage thresholds")
        if voltage <= self.trip:
            self.enabled = False
        elif voltage >= self.recover:
            self.enabled = True
        return self.enabled


def slip_detected(gyro_yaw, encoder_yaw, gyro_variance, encoder_variance, threshold=9):
    variance = gyro_variance + encoder_variance
    if variance <= 0 or min(gyro_variance, encoder_variance) < 0 or threshold <= 0:
        raise ValueError("invalid slip detector uncertainty")
    nis = (gyro_yaw - encoder_yaw) ** 2 / variance
    return bool(nis > threshold), float(nis)


def localization_healthy(covariance, fix_age, max_position_sigma=0.4, max_heading_sigma=0.5, max_age=2):
    covariance = np.asarray(covariance)
    if covariance.shape[0] < 3 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("pose covariance required")
    finite = np.all(np.isfinite(covariance))
    if not finite:
        return False
    position = np.linalg.eigvalsh(covariance[:2, :2]).max()
    return bool(
        position <= max_position_sigma**2
        and covariance[2, 2] <= max_heading_sigma**2
        and 0 <= fix_age <= max_age
        and np.linalg.eigvalsh(covariance).min() >= -1e-12
    )


def inhibit_reason(
    emergency=False,
    overcurrent=False,
    overtemperature=False,
    undervoltage=False,
    collision=False,
    stale_command=False,
    localization_lost=False,
):
    conditions = [
        (emergency, "emergency"),
        (overcurrent, "overcurrent"),
        (overtemperature, "overtemperature"),
        (undervoltage, "undervoltage"),
        (collision, "collision"),
        (stale_command, "watchdog"),
        (localization_lost, "localization"),
    ]
    return next((name for active, name in conditions if active), None)


def rearm_allowed(
    speed,
    current,
    clearance,
    emergency_released,
    healthy_dwell,
    speed_threshold=0.02,
    current_threshold=0.2,
    required_clearance=0.1,
    required_dwell=0.5,
):
    values = (speed, current, clearance, healthy_dwell)
    if not all(np.isfinite(v) for v in values):
        return False
    return bool(
        emergency_released
        and abs(speed) <= speed_threshold
        and abs(current) <= current_threshold
        and clearance >= required_clearance
        and healthy_dwell >= required_dwell
    )


def bounded_command(command, limits):
    command, limits = np.asarray(command), np.asarray(limits)
    if command.shape != limits.shape or np.any(limits <= 0) or np.any(~np.isfinite(limits)):
        raise ValueError("matching finite positive command limits required")
    if np.any(~np.isfinite(command)):
        return np.zeros_like(command, dtype=float), False
    return np.clip(command, -limits, limits), True


@dataclass(frozen=True)
class Disturbance:
    start: float
    end: float
    kind: str
    value: float

    def __post_init__(self):
        if not np.isfinite(self.start + self.end + self.value) or self.start < 0 or self.end <= self.start:
            raise ValueError("invalid disturbance interval")
        if self.kind not in {
            "grade",
            "left_traction",
            "right_traction",
            "gyro_bias",
            "encoder_dropout",
            "position_dropout",
            "emergency",
            "load",
        }:
            raise ValueError("unknown disturbance kind")

    def active(self, time):
        return self.start <= time < self.end


def disturbance_value(events, kind, time, default=0):
    active = [event.value for event in events if event.kind == kind and event.active(time)]
    if len(active) > 1:
        raise ValueError("overlapping disturbances of the same kind")
    return active[0] if active else default
