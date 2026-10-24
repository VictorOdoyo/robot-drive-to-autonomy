import json

from robotlab.batch import run_batch
from robotlab.cases import Scenario


def test_batch_summaries_match_report_metrics(tmp_path):
    case = Scenario(name="short", duration=0.1, width=3, height=3, goal=(2.3, 0.6), shelves=())
    messages = []
    summary = run_batch([case], tmp_path, messages.append)
    measured = json.loads((tmp_path / "01" / "metrics.json").read_text())
    assert summary["short"]["energy_residual_j"] == measured["energy_residual_j"]
    assert messages == ["[1/1] short"]


def test_batch_rejects_duplicate_names_before_running(tmp_path):
    import pytest

    case = Scenario(duration=0.1)
    with pytest.raises(ValueError, match="unique"):
        run_batch([case, case], tmp_path)
