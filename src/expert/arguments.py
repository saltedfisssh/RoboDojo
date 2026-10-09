"""Common CLI validation; keep it independent of simulator dependencies."""

import argparse


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
    parser.add_argument("--task", required=True)
    parser.add_argument("--planner", help="custom TaskPlanner factory, module:PlannerClass")
    parser.add_argument("--num", type=positive_int, default=10, help="target number of successful episodes")
    parser.add_argument("--seed-start", type=nonnegative_int, default=100000)
    parser.add_argument("--gpu", type=nonnegative_int, default=0, help="GPU index for rendering and cuRobo")
    parser.add_argument("--no-images", action="store_true", help="skip rendering (success-rate runs)")
    parser.add_argument("--table-randomization", type=probability, default=0.0)
