import numpy as np
import pytest

from robotlab.analysis import mission_metrics
from robotlab.cases import get_case
from robotlab.simulation import run


@pytest.fixture(scope="module")
def baseline():
    return run(get_case("commissioning"))


def test_warehouse_completes_with_electrical_and_geometric_bounds():
    result = run(get_case("warehouse"))
    metrics = mission_metrics(result)
    assert metrics["completed"]
    assert metrics["goal_error_m"] < 0.15
    assert metrics["minimum_clearance_m"] > 0.12
    assert metrics["position_rmse_m"] < 0.06
    assert metrics["peak_current_a"] < 8.5
    assert abs(metrics["energy_residual_j"]) < 0.002


@pytest.mark.parametrize(
    "name", ["grade-load", "gyro-bias", "position-outage", "hot-windings", "sensor-latency", "low-charge"]
)
def test_stress_experiments_preserve_clearance_and_expose_the_injected_effect(name, baseline):
    result = run(get_case(name))
    metrics = mission_metrics(result)
    reference = mission_metrics(baseline)
    assert metrics["completed"]
    assert metrics["minimum_clearance_m"] > 0.1
    assert metrics["goal_error_m"] < 0.2
    assert metrics["peak_current_a"] < 8.5
    assert abs(metrics["energy_residual_j"]) < 0.01
    if name == "grade-load":
        assert metrics["external_work_j"] > 10
        assert metrics["dissipated_j"] > reference["dissipated_j"]
    elif name == "gyro-bias":
        assert metrics["heading_rmse_rad"] > reference["heading_rmse_rad"] * 2
    elif name == "position-outage":
        reasons = [event["reason"] for event in result["journal"]]
        assert "localization_timeout" in reasons and reasons[-1] is None
    elif name == "hot-windings":
        assert metrics["peak_current_a"] < reference["peak_current_a"]
        assert np.max(result["current_reference"]) < 2
    elif name == "sensor-latency":
        assert metrics["peak_current_a"] > reference["peak_current_a"]
    elif name == "low-charge":
        assert result["bus_voltage"].min() < baseline["bus_voltage"].min() - 3
