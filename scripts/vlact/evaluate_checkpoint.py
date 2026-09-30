"""Task-balanced, held-out VLAct action/VQA validation; never a simulator score."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlact-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vqa-per-task", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Continue a matching partial frame report")
    args = parser.parse_args()
    if min(args.frames_per_episode, args.batch_size) < 1 or args.vqa_per_task < 0:
        parser.error("Frame/batch counts must be positive; VQA count must be nonnegative")
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.vlact_root.resolve()))
    from examples.RoboDojo.data import RoboDojoDataset, collate
    from examples.RoboDojo.evaluate_subtasks import ActionErrors, check_split
    from examples.RoboDojo.prompt_audit import PromptAudit
    from examples.RoboDojo.vqa import AnnotationVQADataset, build_vqa_inputs
    import numpy as np
    from omegaconf import OmegaConf
    from starVLA.model.framework import build_framework
    import torch
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm

    started = time.monotonic()
    torch.manual_seed(42)
    np.random.seed(42)
    run = args.checkpoint.resolve().parents[1]
    cfg = OmegaConf.load(run / "config.yaml")
    if cfg.datasets.vla_data.include_state or cfg.datasets.vla_data.language_mode != "subtask":
        raise ValueError("Expected subtask-conditioned checkpoint without state input")
    cfg.trainer.enable_gradient_checkpointing = False
    cfg.framework.qwenvl.base_vlm = args.base_vlm
    dataset = RoboDojoDataset(args.cache, split="val")
    check_split(dataset.manifest, json.loads((run / "data_manifest.json").read_text()))
    if json.loads((args.cache / "dataset_statistics.json").read_text()) != json.loads(
        (run / "dataset_statistics.json").read_text()
    ):
        raise ValueError("Training and validation normalization statistics differ")
    indices, selection, base = [], [], 0
    for episode in dataset.episodes:
        # Stratified temporal bin centers cover the whole trajectory, including later subtasks.
        frames = np.unique(
            ((np.arange(args.frames_per_episode) + 0.5) * episode["length"] / args.frames_per_episode).astype(int)
        )
        for frame in frames:
            indices.append(base + int(frame))
            selection.append(
                {"episode": episode["original_episode_index"], "task_index": episode["task_index"], "frame": int(frame)}
            )
        base += episode["length"]
    if not indices:
        raise ValueError("No validation frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selection_path = args.output.with_suffix(".selection.json")
    frames_path = args.output.with_suffix(".frames.jsonl")
    run_path = args.output.with_suffix(".run.json")
    identity = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
        "config_sha256": hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest(),
        "cache": str(args.cache.resolve()),
        "base_vlm": args.base_vlm,
        "frames_per_episode": args.frames_per_episode,
        "batch_size": args.batch_size,
        "vqa_per_task": args.vqa_per_task,
    }
    previous = []
    if args.resume:
        if json.loads(run_path.read_text()) != identity:
            raise ValueError("Partial evaluation identity differs from requested run")
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError("Partial evaluation frame selection differs")
        previous = [json.loads(line) for line in frames_path.read_text().splitlines()]
        if len(previous) > len(selection):
            raise ValueError("Partial report has more frames than the selection")
        for record, expected in zip(previous, selection):
            if {key: record[key] for key in expected} != expected:
                raise ValueError("Partial frame report is not a prefix of the requested selection")
    else:
        if any(path.exists() for path in (frames_path, selection_path, run_path)):
            raise FileExistsError("Partial evaluation exists; use --resume or another output")
        selection_path.write_text(json.dumps(selection, indent=2) + "\n")
        run_path.write_text(json.dumps(identity, indent=2) + "\n")
    model = build_framework(cfg)
    weights = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    model.eval().cuda()
    audit = PromptAudit(model.qwen_vl_interface.processor.tokenizer, args.output.with_suffix(".inputs.jsonl"))
    hook = model.qwen_vl_interface.register_forward_pre_hook(audit, with_kwargs=True)
    total, by_task = ActionErrors(), defaultdict(ActionErrors)
    for record in previous:
        errors = np.asarray(record["normalized_l1_per_dimension"], dtype=np.float64)
        steps = record["valid_action_steps"]
        if errors.shape != (14,) or not np.isfinite(errors).all() or not 1 <= steps <= 50:
            raise ValueError("Invalid partial frame metrics")
        for accumulator in (total, by_task[str(record["task_index"])]):
            accumulator.absolute_sum += errors * steps
            accumulator.valid_steps += steps
            accumulator.frames += 1
    task_names = {str(ep["task_index"]): ep["task"] for ep in dataset.episodes}
    loader = DataLoader(
        Subset(dataset, indices[len(previous) :]),
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate,
        shuffle=False,
    )
    with frames_path.open("a" if args.resume else "w") as frames_file:
        for examples in tqdm(loader, desc="Held-out actions", unit="batch"):
            if audit.batches < 2:
                audit.arm(examples, phase="offline_validation")
            with torch.inference_mode():
                result = model.predict_action(examples=[{"image": e["image"], "lang": e["lang"]} for e in examples])
            predictions = np.asarray(result["normalized_actions"], dtype=np.float32)
            if predictions.shape != (len(examples), 50, 14):
                raise ValueError(f"Unexpected action shape: {predictions.shape}")
            for prediction, example in zip(predictions, examples, strict=True):
                total.add(prediction, example["action"], example["action_valid_mask"])
                by_task[str(example["task_index"])].add(prediction, example["action"], example["action_valid_mask"])
                frame_error = ActionErrors()
                frame_error.add(prediction, example["action"], example["action_valid_mask"])
                frames_file.write(
                    json.dumps(
                        {
                            "episode": example["original_episode_index"],
                            "frame": example["frame_index"],
                            "task_index": example["task_index"],
                            "current_subtask": example["current_subtask"],
                            **frame_error.report(),
                        }
                    )
                    + "\n"
                )
            frames_file.flush()
    hook.remove()
    per_task = {
        key: {"task": task_names[key], **value.report()}
        for key, value in sorted(by_task.items(), key=lambda item: int(item[0]))
    }
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cache": str(args.cache.resolve()),
        "checkpoint_bytes": args.checkpoint.stat().st_size,
        "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
        "config_sha256": hashlib.sha256((run / "config.yaml").read_bytes()).hexdigest(),
        "conditioning": "reviewed_subtask_labels"
        if dataset.manifest["spec"]["annotations_reviewed"]
        else "unreviewed_pseudo_subtask_labels",
        "simulator_evaluation": False,
        "success_rate": None,
        "split": "validation",
        "selection": "temporal bin centers in every held-out episode",
        "frames_per_episode": args.frames_per_episode,
        "batch_size": args.batch_size,
        "resumed_frames": len(previous),
        "complete_held_out_frames": False,
        "episodes": len(dataset.episodes),
        "tasks": len(per_task),
        "micro": total.report(),
        "task_macro_normalized_l1": float(np.mean([v["normalized_l1"] for v in per_task.values()])),
        "per_task": per_task,
        "strict_load": True,
        "include_state": False,
        "action_mask": "current subtask and episode boundaries",
        "normalization": "checkpoint train-only q01/q99; target clipped to [-1,1]",
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    if args.vqa_per_task:
        vqa = AnnotationVQADataset(args.cache, split="val")
        rows_by_task = defaultdict(list)
        for index, row in enumerate(vqa.rows):
            rows_by_task[str(vqa.episodes[row["source_episode_index"]]["task_index"])].append(index)
        vqa_records = []
        for task, rows in tqdm(sorted(rows_by_task.items()), desc="Held-out VQA", unit="task"):
            picks = np.unique(((np.arange(args.vqa_per_task) + 0.5) * len(rows) / args.vqa_per_task).astype(int))
            for pick in picks:
                sample = vqa[rows[int(pick)]]
                inputs, audits = build_vqa_inputs(model.qwen_vl_interface.processor, [sample])
                with torch.inference_mode():
                    loss = model.qwen_vl_interface(**inputs.to(model.device), return_dict=True, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite VQA cross entropy")
                vqa_records.append({"task_index": task, "cross_entropy": float(loss), **audits[0]})
        tokens = sum(r["supervised_tokens"] for r in vqa_records)
        report["vqa"] = {
            "examples": len(vqa_records),
            "tasks": len(rows_by_task),
            "answer_tokens": tokens,
            "answer_token_cross_entropy": sum(r["cross_entropy"] * r["supervised_tokens"] for r in vqa_records)
            / tokens,
            "free_generation_evaluated": False,
            "records": vqa_records,
        }
    report["elapsed_seconds"] = time.monotonic() - started
    report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_task", "vqa"}}, indent=2))


if __name__ == "__main__":
    main()
