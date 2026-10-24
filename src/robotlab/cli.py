"""Command line entry points; all output paths are user-controlled local artifacts."""

import argparse
import json
from dataclasses import replace

from .analysis import mission_metrics
from .cases import experiment_catalog, get_case, load_case
from .reporting import write_report
from .simulation import run


def parser():
    root = argparse.ArgumentParser(
        prog="robotlab", description="Electrical drive to autonomous robot laboratory"
    )
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list built-in experiments")
    execute = commands.add_parser("run", help="simulate one experiment and export a report")
    source = execute.add_mutually_exclusive_group()
    source.add_argument("--case", choices=sorted(experiment_catalog()))
    source.add_argument("--config", help="strict scenario JSON")
    execute.add_argument("--duration", type=float)
    execute.add_argument("--dt", type=float)
    execute.add_argument("--seed", type=int)
    execute.add_argument("--output", default="outputs/run")
    return root


def configured_case(arguments):
    case = load_case(arguments.config) if arguments.config else get_case(arguments.case or "warehouse")
    updates = {
        key: getattr(arguments, key)
        for key in ("duration", "dt", "seed")
        if getattr(arguments, key) is not None
    }
    if "duration" in updates:
        duration = updates["duration"]
        updates["events"] = tuple(
            replace(event, end=min(event.end, duration)) for event in case.events if event.start < duration
        )
    return replace(case, **updates)


def main(argv=None):
    arguments_parser = parser()
    arguments = arguments_parser.parse_args(argv)
    try:
        if arguments.command == "list":
            print(
                json.dumps(
                    {
                        name: {"duration_s": case.duration, "events": len(case.events)}
                        for name, case in experiment_catalog().items()
                    },
                    indent=2,
                )
            )
        elif arguments.command == "run":
            case = configured_case(arguments)
            result = run(case)
            report = write_report(result, arguments.output)
            print(json.dumps({**mission_metrics(result), "report": str(report.resolve())}, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        arguments_parser.error(str(error))
    return 0
