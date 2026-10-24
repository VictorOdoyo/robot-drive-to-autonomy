import json
from pathlib import Path

from .analysis import mission_metrics
from .reporting import write_report
from .simulation import run


def run_batch(cases, directory, progress=None):
    cases = list(cases)
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("batch experiment names must be unique")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary = {}
    for index, case in enumerate(cases):
        if progress:
            progress(f"[{index + 1}/{len(cases)}] {case.name}")
        # Stable numeric folder names keep display names out of filesystem paths.
        result = run(case)
        report = write_report(result, directory / f"{index + 1:02d}")
        summary[case.name] = {**mission_metrics(result), "report": report.relative_to(directory).as_posix()}
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary
