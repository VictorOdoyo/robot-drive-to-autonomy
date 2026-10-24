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
