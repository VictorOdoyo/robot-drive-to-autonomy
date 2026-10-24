import numpy as np
import pytest

from robotlab import reporting as m


def test_provenance_hash_authenticates_actual_configuration():
    from robotlab.cases import Scenario

    first = m.provenance(Scenario().to_dict(), {"mass": 18})
    second = m.provenance(Scenario().to_dict(), {"mass": 19})
    assert first["configuration_sha256"] != second["configuration_sha256"]
    assert len(first["configuration_sha256"]) == 64
    assert np.isfinite(float(first["package_version"].split(".")[0]))
    assert first["package_version"] == "0.1.0"
    assert first == m.provenance(Scenario().to_dict(), {"mass": 18})


def test_archive_validation_rejects_nonfinite_trace():
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    assert m.validate_result(result) is result
    result["state"][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        m.validate_result(result)
