"""Unified data generation CLI with lazy command loading."""

import argparse
from importlib import import_module
import json
from pathlib import Path
import subprocess

from src.expert.arguments import help_parser

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

    parser = help_parser(
        "tasks",
        "List official tasks, planner status and implementation notes without starting Isaac.",
        epilog="Examples:\n  %(prog)s\n  %(prog)s --implemented --json\n\n"
        "Status: available = registered as ready; experimental = needs further validation;\n"
        "unimplemented = no planner. An implemented planner is not necessarily qualified.",
    )
    parser.add_argument(
        "--implemented", action="store_true", help="show only tasks with available or experimental planners"
    )
    parser.add_argument("--json", action="store_true", help="print a JSON array instead of the task/status/notes table")
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
    parser = help_parser(
        "",
        "Generate expert demonstrations from official RoboDojo task layouts.\n"
        "Only episodes passing the official task reward count as successful data.",
        epilog="Commands:\n"
        + "\n".join(f"  {name:10} {description}" for name, description in COMMANDS.items())
        + "\n\nQuick start (run from the repository root):\n"
        "  %(prog)s tasks --implemented\n"
        "  %(prog)s collect --task stack_blocks --num 3 --no-images --seed-start 0\n"
        "  %(prog)s batch --task stack_blocks --num 100 --out data/expert/raw/stack_blocks\n"
        "  %(prog)s batch --task stack_blocks --num 100 --format lerobot-v3 --out data/expert/lerobot/stack_blocks\n"
        "  %(prog)s export --input data/expert/raw/stack_blocks --output data/expert/lerobot/stack_blocks\n"
        "\nHelp:\n"
        "  %(prog)s collect --help\n"
        "  %(prog)s dev run --help\n"
        "  scripts/expert.sh is a compatibility alias with the same commands.\n"
        "\nEnvironments and paths:\n"
        "  ROBODOJO_VENV selects the simulator venv (default: <repo>/.venv).\n"
        "  LEROBOT_VENV selects the export venv (default: <repo>/.venv-lerobot).\n"
        "  LEROBOT_PYTHON overrides the export interpreter, including during dev qualify.\n"
        "  Shell launchers resolve relative paths from the repository root.\n"
        "  Expert command help needs only Python 3.11+; neither venv nor Isaac is needed.\n"
        "\nSetup: bash scripts/install-uv.sh --with-export && bash scripts/init_assets-uv.sh\n"
        "Docs: docs/expert/README.md; planner development: docs/expert/DEVELOPMENT.md",
    )
    parser.add_argument("command", nargs="?", choices=COMMANDS, metavar="COMMAND", help="workflow step listed below")
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, metavar="ARGS", help="command options; use COMMAND --help for details"
    )
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
