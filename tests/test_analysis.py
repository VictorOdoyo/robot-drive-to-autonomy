import numpy as np
import pytest

from robotlab import analysis as m


def test_metrics_use_true_pose_and_energy_ledger():
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.3, width=3, height=3, start=(0.6, 0.6, 0), goal=(2.3, 0.6), shelves=()))
    values = m.mission_metrics(result)
    assert values["distance_m"] > 0
    assert values["minimum_clearance_m"] > 0
    assert values["energy_residual_j"] == pytest.approx(result["energy_residual_j"])
    assert np.isfinite(values["position_rmse_m"])


def test_consistency_detects_underestimated_covariance():
    result = {
        "estimate": np.array([[1.0, 0, 0, 0]]),
        "state": np.zeros((1, 18)),
        "covariance": np.array([np.eye(4) * 0.001]),
    }
    summary = m.consistency(result)
    assert summary["mean_nees"] == pytest.approx(1000)
    assert summary["coverage"] == 0


def test_analytic_first_order_settling():
    time = np.linspace(0, 10, 10001)
    response = 1 - np.exp(-time)
    summary = m.step_metrics(time, response, 1)
    assert summary["settling_s"] == pytest.approx(-np.log(0.02), abs=0.002)
    assert summary["overshoot_fraction"] == 0
