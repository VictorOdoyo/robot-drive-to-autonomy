from itertools import pairwise

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


def test_warehouse_route_has_clearance():
    from robotlab.geometry import circle_clearance

    case = m.Scenario()
    points, _, _, cost = m.plan_route(case)
    assert np.allclose(points[0], case.start[:2])
    assert np.allclose(points[-1], case.goal)
    assert cost > 0
    for a, b in pairwise(points):
        assert all(circle_clearance(p, case.radius, case.obstacles()) > 0.15 for p in np.linspace(a, b, 30))


def test_all_experiments_have_portable_valid_disturbances():
    import json

    cases = m.experiment_catalog()
    assert "warehouse" in cases and "emergency-stop" in cases
    for name, case in cases.items():
        assert case.name == name
        assert m.from_dict(json.loads(json.dumps(case.to_dict()))) == case


@pytest.mark.parametrize(
    "values",
    [
        {"seed": 1.5},
        {"seed": True},
        {"name": ""},
        {"margin": np.nan},
        {"sensor_latency": np.inf},
        {"shelves": ((0, 0, np.nan, 1),)},
    ],
)
def test_malformed_scenario_values(values):
    with pytest.raises(ValueError):
        m.Scenario(**values)
