import numpy as np
import pytest

from robotlab import sensors as m


def test_encoder_rollover():
    encoder = m.Encoder(1024, 16)
    a, b = encoder.count(-0.01), encoder.count(0.01)
    delta = encoder.increment(a, b)
    assert delta == pytest.approx(0.02, abs=2 * np.pi / 1024)
    assert np.isfinite(delta)
