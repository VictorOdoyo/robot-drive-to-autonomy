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
