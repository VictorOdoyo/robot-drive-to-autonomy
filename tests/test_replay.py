import numpy as np

from robotlab.cases import Scenario
from robotlab.cli import replay_command
from robotlab.reporting import load_archive, save_archive, verify_report
from robotlab.simulation import run


def test_report_replay_preserves_physics_and_source(tmp_path):
    result = run(Scenario(duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=()))
    source = save_archive(result, tmp_path / "source.npz")
    original = load_archive(source)
    summary = replay_command(source, tmp_path / "replay")
    replayed = load_archive(tmp_path / "replay" / "mission.npz")
    assert np.array_equal(original["state"], replayed["state"])
    assert replayed["provenance"] == original["provenance"]
    assert summary["duration_s"] == 0.1
    assert verify_report(tmp_path / "replay")
