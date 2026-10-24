import numpy as np

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
