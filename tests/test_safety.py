import numpy as np
import pytest

from robotlab import safety as m


def test_fault_latch_requires_healthy_dwell():
    latch = m.FaultLatch(0.5)
    assert not latch.update(0, "overcurrent")
    assert not latch.update(0.1, reset=True)
    assert not latch.update(0.4, reset=True)
    assert latch.update(0.7, reset=True)
    assert np.isfinite(latch.dwell)
    assert latch.dwell == pytest.approx(0.5)
