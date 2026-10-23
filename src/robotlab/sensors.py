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
