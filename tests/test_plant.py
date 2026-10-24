import numpy as np
import pytest

from robotlab import plant as m


def test_plant_initial_state():
    state = m.initial_state(pose=(1, 2, 0.3), soc=0.7)
    assert np.allclose(state[6:9], [1, 2, 0.3])
    assert state[9] == pytest.approx(0.7)
    assert np.allclose(state[11:14], 25)
