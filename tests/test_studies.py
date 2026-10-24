import numpy as np
import pytest

from robotlab import studies as m


def test_sampling_and_delay_stability_regions():
    periods = [0.0005, 0.001, 0.002]
    bandwidths = [100, 500, 1500]
    direct = m.current_stability(periods, bandwidths)
    delayed = m.current_stability(periods, bandwidths, delayed=True)
    assert np.all(direct[:, 0] < 1)
    assert np.any((direct < 1) & (delayed > 1))
    assert direct.shape == (3, 3)
    assert np.max(delayed) > pytest.approx(1).expected
