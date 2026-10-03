"""Planner annotations and an offline extension protocol for LLM/VLM providers."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AnnotationContext:
    metadata: dict
    segments: list[dict]
    raw_episode: str
    fps: int
    num_frames: int


class AnnotationProvider(Protocol):
    def __call__(self, context: AnnotationContext) -> dict:
        """Return {persistent: [LeRobot rows], events: {frame_index: [LeRobot rows]}}.

        Rows extend the planner subtask timeline. Providers can inspect raw_episode's
        images. Calls, credentials, batching, and model selection belong to the provider.
        No remote service is called by the collector/exporter itself.
        """
        ...


def segments_from_frames(annotations):
    spans = []
    for i, ann in enumerate(annotations):
        if not spans or ann["segment_id"] != spans[-1]["segment_id"]:
            if spans:
                spans[-1]["end_frame"] = i
            spans.append({**ann, "start_frame": i})
        else:
            # Multi-waypoint skills (pouring, turning) update their target while
            # executing. The segment's goal is its final target, not the first.
            spans[-1]["goal"] = ann.get("goal", {})
    if spans:
        spans[-1]["end_frame"] = i + 1
    return spans


def persistent_rows(segments, fps, start=0, end=None):
    return [
        {
            "role": "assistant",
            "content": s["text"],
            "style": "subtask",
            "timestamp": max(0, s["start_frame"] - start) / fps,
            "camera": None,
            "tool_calls": None,
        }
        for s in segments
        if s["end_frame"] > start and (end is None or s["start_frame"] < end)
    ]


def load_provider(spec):
    import importlib

    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def slice_persistent(rows, start_s, end_s):
    """Carry active state at a crop boundary, then rebase subsequent changes."""
    active, changes = {}, []
    for row in sorted(rows, key=lambda r: r["timestamp"]):
        if row["timestamp"] <= start_s:
            active[(row["role"], row["style"], row.get("camera"))] = {**row, "timestamp": 0.0}
        elif row["timestamp"] < end_s:
            changes.append({**row, "timestamp": row["timestamp"] - start_s})
    return [*active.values(), *changes]


def validate_additions(additions, context, camera_keys):
    """Validate providers before writing videos or mutating language columns."""
    import math

    from lerobot.datasets.language import column_for_style, validate_camera_field

    if not isinstance(additions, dict) or set(additions) - {"persistent", "events"}:
        raise ValueError("Provider must return a dict with persistent and/or events")

    def check(row, column):
        if row.get("role") not in {"assistant", "user", "system", "tool"}:
            raise ValueError("Language row requires a valid role")
        if column_for_style(row.get("style")) != column:
            raise ValueError("Language style is in the wrong column")
        validate_camera_field(row.get("style"), row.get("camera"))
        if row.get("camera") is not None and row["camera"] not in camera_keys:
            raise ValueError("Language row references an unknown camera")
        if row.get("content") is None and not row.get("tool_calls"):
            raise ValueError("Language row requires content or tool_calls")
        if row.get("style") == "subtask":
            raise ValueError("Provider must not replace the planner subtask timeline")

    for row in additions.get("persistent", []):
        check(row, "language_persistent")
        timestamp = row.get("timestamp")
        if not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp):
            raise ValueError("Persistent timestamp must be finite")
        if not 0 <= timestamp < context.num_frames / context.fps:
            raise ValueError("Persistent timestamp is outside the episode")
    for frame, rows in additions.get("events", {}).items():
        if str(int(frame)) != str(frame) or not 0 <= int(frame) < context.num_frames:
            raise ValueError("Event frame must be an integer inside the episode")
        for row in rows:
            check(row, "language_events")
            if "timestamp" in row:
                raise ValueError("Events attach to a frame and must not contain timestamp")
