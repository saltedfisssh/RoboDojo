"""Report known unsupported default-sampler semantics before starting Isaac.

This is a conservative compatibility check, not a proof that a scene is valid.
Task-local sampler adapters can implement the missing semantics explicitly.
"""

from pathlib import Path


def audit_layout(task, root=None):
    import yaml

    root = Path(root) if root else Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / f"task/RoboDojo/config/{task}.yml").read_text())
    index = yaml.safe_load((root / "task/RoboDojo/config/_task.yml").read_text())
    settings = {**index.get("common", {}), **index.get("tasks", {}).get(task, {})}
    scene = settings.get("scene_config", "default")
    issues = []
    if scene != "default":
        issues.append(f"scene_config={scene}: default sampler loads scene/default.yml")
    kinds = {"Rigid", "Geometry", "Articulation", "Dynamic", "Garment", "Fluid"}
    if config.get("Clutter"):
        issues.append("Clutter: default sampler does not generate distractors")
    for kind in kinds:
        if kind not in ("Rigid", "Geometry") and config.get(kind):
            issues.append(f"{kind}: task-specific initialization and state adapter require validation")
        for index, group in enumerate(config.get(kind, []) or []):
            location = f"{kind}[{index}]"
            mode = group.get("select_mode", {}).get("mode", "unique")
            if mode not in ("hierarchical", "unique", "same", "allow_duplicate"):
                issues.append(f"{location}.select_mode={mode}: unsupported")
            if group.get("dynamic_layout"):
                issues.append(f"{location}.dynamic_layout: unsupported")
            for entry in group.get("category", []):
                values = {**group.get("common", {}), **entry}
                label = f"{location}/{entry.get('name')}"
                if str(values.get("relative_plane", "Table")).lower() not in ("table", "ground"):
                    issues.append(f"{label}.relative_plane: object-relative placement unsupported")
                if values.get("place_tag"):
                    issues.append(f"{label}.place_tag: functional placement orientation unsupported")
                if values.get("scale", [1, 1, 1]) != [1, 1, 1]:
                    issues.append(f"{label}.scale: scaled bounds unsupported")
                for key in ("xlim", "ylim", "zlim"):
                    limits = values.get(key)
                    if limits is None:
                        continue
                    if len(limits) != 2 or any(isinstance(value, list | dict) for value in limits):
                        issues.append(f"{label}.{key}: disjoint or malformed range unsupported")
                    elif key == "zlim" and limits[0] != limits[1]:
                        issues.append(f"{label}.zlim: sampled height range unsupported")
                if values.get("check_mode", "bbox") != "bbox":
                    issues.append(f"{label}.check_mode: non-bbox placement checks unsupported")
    return {"task": task, "scene": scene, "default_sampler_compatible": not issues, "issues": sorted(set(issues))}
