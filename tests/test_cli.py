import json

from robotlab.cli import main


def test_cli_list_and_short_run(capsys, tmp_path):
    assert main(["list"]) == 0
    assert "warehouse" in json.loads(capsys.readouterr().out)
    assert main(["run", "--case", "commissioning", "--duration", ".1", "--output", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["duration_s"] == 0.1
    assert (tmp_path / "mission.npz").is_file()
