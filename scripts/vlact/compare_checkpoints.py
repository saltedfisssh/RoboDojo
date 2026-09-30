"""Compare paired held-out validation reports; write JSON, CSV and a figure."""

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--earlier", type=Path, required=True)
    parser.add_argument("--later", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    earlier, later = [json.loads(p.read_text()) for p in (args.earlier, args.later)]
    for key in ("cache", "config_sha256", "conditioning", "split", "frames_per_episode", "batch_size"):
        if earlier[key] != later[key]:
            raise ValueError(f"Incomparable evaluation setting: {key}")
    if args.earlier.with_suffix(".selection.json").read_text() != args.later.with_suffix(".selection.json").read_text():
        raise ValueError("Frame selections differ")
    vqa_keys = ("task_index", "source_episode_index", "frame_index", "camera", "question", "answer")
    for first, second in zip(earlier["vqa"]["records"], later["vqa"]["records"], strict=True):
        if any(first[k] != second[k] for k in vqa_keys):
            raise ValueError("VQA samples differ")
    rows = []
    for task, first in earlier["per_task"].items():
        second = later["per_task"][task]
        if (first["frames"], first["valid_action_steps"]) != (second["frames"], second["valid_action_steps"]):
            raise ValueError("Action masks differ")
        rows.append(
            {
                "task_index": task,
                "task": first["task"],
                "earlier_l1": first["normalized_l1"],
                "later_l1": second["normalized_l1"],
                "delta_later_minus_earlier": second["normalized_l1"] - first["normalized_l1"],
            }
        )
    summary = {
        "earlier_checkpoint": earlier["checkpoint"],
        "later_checkpoint": later["checkpoint"],
        "paired_action_frames": earlier["micro"]["frames"],
        "paired_episodes": earlier["episodes"],
        "paired_vqa_examples": earlier["vqa"]["examples"],
        "tasks": len(rows),
        "tasks_improved": sum(r["delta_later_minus_earlier"] < 0 for r in rows),
        "earlier_macro_l1": earlier["task_macro_normalized_l1"],
        "later_macro_l1": later["task_macro_normalized_l1"],
        "earlier_micro_l1": earlier["micro"]["normalized_l1"],
        "later_micro_l1": later["micro"]["normalized_l1"],
        "earlier_vqa_ce": earlier["vqa"]["answer_token_cross_entropy"],
        "later_vqa_ce": later["vqa"]["answer_token_cross_entropy"],
        "simulator_evaluation": False,
        "success_rate": None,
    }
    summary["macro_l1_relative_reduction_percent"] = 100 * (1 - summary["later_macro_l1"] / summary["earlier_macro_l1"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "per_task.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array([r["earlier_l1"] for r in rows])
    y = np.array([r["later_l1"] for r in rows])
    limit = float(max(x.max(), y.max())) * 1.12
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    ax.plot([0, limit], [0, limit], color="0.65", linestyle="--", linewidth=1)
    improved = y < x
    ax.scatter(x[improved], y[improved], color="#167d8d", s=42, label=f"Improved: {improved.sum()} tasks")
    ax.scatter(x[~improved], y[~improved], color="#d45b36", s=42, label=f"Worse: {(~improved).sum()} tasks")
    for index in np.argsort(np.abs(y - x))[-6:]:
        ax.annotate(
            rows[index]["task_index"], (x[index], y[index]), xytext=(4, 3), textcoords="offset points", fontsize=8
        )
    ax.legend(loc="upper left", frameon=False)
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="30k normalized action L1",
        ylabel="40k normalized action L1",
        title="35 tasks · paired pseudo-label-conditioned validation",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    fig.savefig(args.output_dir / "action_l1_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
