import numpy as np
import pytest

from robotlab import cases as m


def test_scenario_validation_and_boundary_geometry():
    case = m.Scenario()
    assert len(case.obstacles()) == 6
    assert case.dt == pytest.approx(0.002)
    assert np.isfinite(case.duration)
    with pytest.raises(ValueError):
        m.Scenario(dt=0.01)


def test_portable_scenario_roundtrip():
    import json

    case = m.Scenario()
    restored = m.from_dict(json.loads(json.dumps(case.to_dict())))
    assert restored == case
    with pytest.raises(ValueError):
        m.from_dict({"surprise": 1})
