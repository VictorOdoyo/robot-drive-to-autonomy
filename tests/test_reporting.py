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


def test_archive_contains_no_python_object_arrays(tmp_path):
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    path = m.save_archive(result, tmp_path / "mission.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        assert archive["state"].shape[1] == 18


def test_archive_roundtrip(tmp_path):
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    restored = m.load_archive(m.save_archive(result, tmp_path / "mission.npz"))
    assert np.array_equal(restored["state"], result["state"])
    assert restored["energy_residual_j"] == pytest.approx(result["energy_residual_j"])


def test_csv_columns_and_units(tmp_path):
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    path = m.export_csv(result, tmp_path / "trace.csv")
    data = np.genfromtxt(path, delimiter=",", names=True)
    assert data["x_m"][-1] == pytest.approx(result["state"][-1, 6])
    assert data["bus_v"][0] == pytest.approx(result["bus_voltage"][0])


def test_mission_plot_renders_from_trace(tmp_path):
    from robotlab.cases import Scenario
    from robotlab.simulation import run

    result = run(Scenario(duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    path = m.plot_mission(result, tmp_path / "mission.png")
    assert path.read_bytes().startswith(b"\x89PNG")
