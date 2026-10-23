from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Encoder:
    counts_per_revolution: int = 2048
    bits: int = 16

    def __post_init__(self):
        if self.counts_per_revolution <= 0 or not 2 <= self.bits <= 32:
            raise ValueError("invalid encoder resolution")

    def count(self, angle):
        return int(np.rint(angle * self.counts_per_revolution / (2 * np.pi))) % (1 << self.bits)

    def increment(self, previous, current):
        modulus = 1 << self.bits
        difference = (current - previous + modulus // 2) % modulus - modulus // 2
        if abs(difference) == modulus // 2:
            raise ValueError("ambiguous half-range rollover")
        return difference * 2 * np.pi / self.counts_per_revolution


def encoder_speed_variance(encoder, dt):
    if dt <= 0:
        raise ValueError("positive sampling period required")
    quantum = 2 * np.pi / encoder.counts_per_revolution
    return quantum**2 / (6 * dt**2)


def count_window_speed(times, unwrapped_counts, counts_per_revolution):
    times, counts = np.asarray(times), np.asarray(unwrapped_counts)
    if (
        len(times) < 2
        or counts.shape != times.shape
        or np.any(np.diff(times) <= 0)
        or counts_per_revolution <= 0
    ):
        raise ValueError("ordered count-time pairs required")
    centered = times - times.mean()
    slope = centered @ (counts - counts.mean()) / (centered @ centered)
    return float(slope * 2 * np.pi / counts_per_revolution)


def gyro_measurement(yaw_rate, bias, noise_std, rng):
    if noise_std < 0:
        raise ValueError("negative noise standard deviation")
    return float(yaw_rate + bias + rng.normal(0, noise_std))


def bias_step(bias, spectral_density, dt, rng):
    if spectral_density < 0 or dt < 0:
        raise ValueError("invalid bias random walk")
    return float(bias + rng.normal(0, np.sqrt(spectral_density * dt)))


def adc(value, full_scale, bits, offset=0):
    if full_scale <= 0 or not 2 <= bits <= 24:
        raise ValueError("invalid ADC range")
    maximum = (1 << (bits - 1)) - 1
    raw = np.clip(np.rint((np.asarray(value) + offset) / full_scale * maximum), -maximum, maximum)
    return raw * full_scale / maximum


class DelayQueue:
    def __init__(self, latency=0.0):
        if latency < 0:
            raise ValueError("negative latency")
        self.latency = latency
        self.queue = []
        self.last_stamp = -np.inf

    def push(self, stamp, value):
        if not np.isfinite(stamp) or stamp < self.last_stamp:
            raise ValueError("measurement timestamps must be ordered")
        self.queue.append((stamp, value))
        self.last_stamp = stamp

    def ready(self, time):
        count = 0
        while count < len(self.queue) and self.queue[count][0] + self.latency <= time + 1e-12:
            count += 1
        ready, self.queue = self.queue[:count], self.queue[count:]
        return ready


def valid_stamp(stamp, now, last_stamp, max_age):
    if max_age < 0:
        raise ValueError("negative allowed age")
    return bool(np.isfinite(stamp) and stamp <= now and stamp > last_stamp and now - stamp <= max_age)


def ray_range(origin, angle, rectangles, max_range=10.0):
    if max_range <= 0:
        raise ValueError("positive range required")
    direction = np.array([np.cos(angle), np.sin(angle)])
    nearest = max_range
    for rectangle in rectangles:
        low, high = 0.0, max_range
        for axis in range(2):
            if abs(direction[axis]) < 1e-14:
                if not rectangle[axis] <= origin[axis] <= rectangle[axis + 2]:
                    high = -1.0
                    break
            else:
                a = (rectangle[axis] - origin[axis]) / direction[axis]
                b = (rectangle[axis + 2] - origin[axis]) / direction[axis]
                low, high = max(low, min(a, b)), min(high, max(a, b))
        if high >= low:
            nearest = min(nearest, low)
    return float(nearest)


def lidar_scan(pose, angles, rectangles, max_range=10, noise_std=0, rng=None, extrinsic=(0, 0, 0)):
    from .geometry import compose

    sensor = compose(pose, extrinsic)
    ranges = np.array([ray_range(sensor[:2], sensor[2] + a, rectangles, max_range) for a in angles])
    if noise_std < 0:
        raise ValueError("negative range noise")
    if noise_std:
        if rng is None:
            raise ValueError("explicit random generator required")
        ranges = np.clip(ranges + rng.normal(0, noise_std, len(ranges)), 0, max_range)
    return ranges


def dropout_mask(times, bursts=(), probability=0, rng=None):
    times = np.asarray(times)
    if not 0 <= probability <= 1:
        raise ValueError("invalid dropout probability")
    keep = np.ones(times.shape, dtype=bool)
    for start, end in bursts:
        if end < start:
            raise ValueError("reversed dropout interval")
        keep &= ~((times >= start) & (times < end))
    if probability:
        if rng is None:
            raise ValueError("explicit random generator required")
        keep &= rng.random(times.shape) >= probability
    return keep
