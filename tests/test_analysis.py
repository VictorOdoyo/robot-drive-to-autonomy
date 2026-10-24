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
