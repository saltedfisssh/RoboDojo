"""Unified data generation CLI with lazy command loading."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import subprocess

COMMANDS = {
    "collect": "collect successful episodes in one Isaac process",
    "batch": "collect with multiple Isaac workers (resumable)",
    "export": "export expert HDF5 to LeRobot v3 and Argus labels",
    "restore": "restore a frame and replay or evaluate a policy",
    "probe": "inspect sampled layouts in Isaac",
    "dev": "scaffold, inspect, run and qualify independently developed planners",
    "tasks": "list official tasks and planner implementation status",
    "legacy": "run the old fixed-layout experimental tools",
}


def list_tasks(argv):
    from src.expert.registry import task_inventory

    parser = argparse.ArgumentParser(prog="generate_data tasks", description=COMMANDS["tasks"])
    parser.add_argument("--implemented", action="store_true", help="include available and experimental planners")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    rows = task_inventory()
    if args.implemented:
        rows = [row for row in rows if row["planner"]]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            print(f"{row['task']:40} {row['status']:14} {row['note']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="generate_data",
        description="RoboDojo expert demonstration generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Commands:\n" + "\n".join(f"  {name:10} {description}" for name, description in COMMANDS.items()),
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "tasks":
        return list_tasks(args.arguments)
    if args.command == "legacy":
        script = Path(__file__).resolve().parents[2] / "scripts/datagen-uv.sh"
        return subprocess.call(["bash", str(script), *args.arguments])
    return import_module(f"src.expert.{args.command}").main(args.arguments)
