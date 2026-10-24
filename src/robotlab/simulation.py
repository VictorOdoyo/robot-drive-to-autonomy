import numpy as np


class PeriodicClock:
    def __init__(self, period):
        if period <= 0:
            raise ValueError("positive clock period required")
        self.period, self.tick = period, 0

    def due(self, time):
        scheduled = self.tick * self.period
        if time + 1e-11 >= scheduled:
            self.tick += 1
            return True
        return False


def simulation_times(case):
    from .numerics import timeline

    clocks = [np.arange(0, case.duration, period) for period in (0.01, 0.02, 0.05, 0.2)]
    events = [value for event in case.events for value in (event.start, event.end)]
    candidates = timeline(case.duration, case.dt, np.r_[*clocks, events])
    merged = [candidates[0]]
    for value in candidates[1:]:
        if value - merged[-1] > 1e-11:
            merged.append(value)
    merged[-1] = case.duration
    for event in events:
        index = int(np.argmin(abs(np.asarray(merged) - event)))
        merged[index] = event
    return np.asarray(merged)


class SensorSuite:
    def __init__(self, case):
        from .sensors import DelayQueue, Encoder

        self.rng = np.random.default_rng(case.seed)
        self.encoder = Encoder()
        self.queue = DelayQueue(case.sensor_latency)
        self.counts = None
        self.last_time = None
        self.rates = np.zeros(2)
        self.age_stamp = -np.inf
        self.position_noise = 0.025
        self.gyro_noise = 0.008

    def sample_wheels(self, state, time, dropout=False):
        counts = [self.encoder.count(angle) for angle in state[14:16]]
        if self.counts is not None:
            dt = time - self.last_time
            increments = np.array(
                [self.encoder.increment(a, b) for a, b in zip(self.counts, counts, strict=True)]
            )
            if not dropout:
                self.queue.push(time, (increments / dt, dt))
        self.counts, self.last_time = counts, time
        for stamp, (rates, dt) in self.queue.ready(time):
            alpha = 1 - np.exp(-dt / 0.03)
            self.rates += alpha * (rates - self.rates)
            self.age_stamp = stamp
        return self.rates.copy()

    def gyro(self, state, bias=0.015):
        from .sensors import gyro_measurement

        return gyro_measurement(state[5], bias, self.gyro_noise, self.rng)

    def position(self, state):
        return state[6:8] + self.rng.normal(0, self.position_noise, 2)
