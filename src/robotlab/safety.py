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
