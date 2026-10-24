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
    replay = commands.add_parser("replay", help="regenerate a report from its recorded trace")
    replay.add_argument("archive")
    replay.add_argument("--output", default="outputs/replay")
    verify = commands.add_parser("verify", help="check report file hashes")
    verify.add_argument("directory")
    batch = commands.add_parser("batch", help="run the experiment catalog")
    batch.add_argument("--cases", nargs="+", choices=sorted(experiment_catalog()))
    batch.add_argument("--output", default="outputs/batch")
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
        elif arguments.command == "batch":
            import sys

            from .batch import run_batch

            names = arguments.cases or list(experiment_catalog())
            summary = run_batch(
                [get_case(name) for name in names],
                arguments.output,
                lambda message: print(message, file=sys.stderr, flush=True),
            )
            print(json.dumps(summary, indent=2))
        elif arguments.command == "replay":
            print(json.dumps(replay_command(arguments.archive, arguments.output), indent=2))
        elif arguments.command == "verify":
            from .reporting import verify_report

            print(json.dumps({"verified": verify_report(arguments.directory)}))
        elif arguments.command == "run":
            case = configured_case(arguments)
            result = run(case)
            report = write_report(result, arguments.output)
            print(json.dumps({**mission_metrics(result), "report": str(report.resolve())}, indent=2))
    except (ValueError, OSError, RuntimeError) as error:
        arguments_parser.error(str(error))
    return 0


def replay_command(archive, output):
    from .reporting import load_archive

    result = load_archive(archive)
    report = write_report(result, output)
    return {"report": str(report.resolve()), **mission_metrics(result)}
