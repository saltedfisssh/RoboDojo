"""Common CLI validation; keep it independent of simulator dependencies."""

import argparse


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Keep examples readable and show only useful, concrete defaults."""

    def _get_help_string(self, action):
        if action.required or action.default is None or isinstance(action.default, bool):
            return action.help
        return super()._get_help_string(action)


def help_parser(command, description, *, epilog=None):
    return argparse.ArgumentParser(
        prog=f"bash scripts/generate_data.sh{(' ' + command) if command else ''}",
        description=description,
        formatter_class=HelpFormatter,
        epilog=epilog,
    )


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def add_collection_arguments(parser):
    task = parser.add_argument_group("Task and target")
    task.add_argument("--task", required=True, help="official task name; list planners with 'tasks --implemented'")
    task.add_argument("--planner", metavar="MODULE:CLASS", help="override the task's registered TaskPlanner factory")
    task.add_argument(
        "--num",
        type=positive_int,
        default=10,
        help="target total successful episodes, including those already saved in --out (not an attempt count)",
    )
    sampling = parser.add_argument_group("Sampling and simulation")
    sampling.add_argument(
        "--seed-start",
        type=nonnegative_int,
        default=100000,
        help="first layout seed; previously attempted seeds in --out are skipped on resume",
    )
    sampling.add_argument(
        "--gpu",
        type=nonnegative_int,
        default=0,
        help="physical GPU index for rendering and cuRobo; do not remap CUDA_VISIBLE_DEVICES",
    )
    sampling.add_argument(
        "--no-images",
        action="store_true",
        help="disable RGB rendering for faster trials; saved episodes cannot be exported to LeRobot",
    )
    sampling.add_argument(
        "--table-randomization",
        type=probability,
        default=0.0,
        metavar="PROBABILITY",
        help="probability of randomizing table/floor materials per episode, from 0 (off) to 1 (always)",
    )
    output = parser.add_argument_group("Recording format")
    output.add_argument(
        "--format",
        choices=("raw", "lerobot-v3"),
        default="raw",
        help="raw: expert HDF5; lerobot-v3: convert and validate each success in the background, then delete its HDF5",
    )
    output.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="skip physical frame checkpoints; automatically enabled by --format lerobot-v3",
    )
    output.add_argument(
        "--export-workers",
        type=positive_int,
        default=1,
        help="concurrent background episode conversions for --format lerobot-v3 (CPU; separate LeRobot environment)",
    )
    output.add_argument(
        "--max-pending-episodes",
        type=positive_int,
        help="pause new attempts at this HDF5 backlog (lerobot-v3 default: 2 x Isaac workers, minimum 2)",
    )
    parser.add_argument("--background-export", action="store_true", help=argparse.SUPPRESS)
