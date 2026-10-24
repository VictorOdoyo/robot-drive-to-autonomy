import json

from robotlab.cli import main


def test_cli_list_and_short_run(capsys, tmp_path):
    assert main(["list"]) == 0
    assert "warehouse" in json.loads(capsys.readouterr().out)
    assert main(["run", "--case", "commissioning", "--duration", ".1", "--output", str(tmp_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["duration_s"] == 0.1
    assert (tmp_path / "mission.npz").is_file()


def test_cli_invalid_timestep_is_an_actionable_error(capsys, tmp_path):
    import pytest

    with pytest.raises(SystemExit) as error:
        main(["run", "--duration", ".1", "--dt", ".1", "--output", str(tmp_path)])
    assert error.value.code == 2
    assert "positive finite scenario" in capsys.readouterr().err
