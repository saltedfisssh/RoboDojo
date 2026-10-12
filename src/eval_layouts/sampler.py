"""Fresh, deterministic initialization layouts for the complete task catalog.

Supports linked model selections, functional support frames, scene overrides,
cloth, fluid, and clutter groups.
The output is the native LayoutManager replay schema; no Eval_Layout is read.
"""

from copy import deepcopy
from functools import cache
import itertools
import json

import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union
from transforms3d.quaternions import qmult, quat2mat
import yaml

from src.expert.layout import (
    GROUND_TOP,
    OBJECTS,
    PHYSICS_OVERRIDES,
    ROOT,
    TASK_CONFIGS,
    LayoutError,
    _choose_models,
    available_indices,
)
from utils.cluttered_generator import ClutteredGenerator


@cache
def metadata(kind, category, index):
    path = OBJECTS / kind / category / f"{index:05d}" / "metadata.json"
    if not path.is_file():
        raise LayoutError(f"Missing configured asset metadata: {path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def position_intervals(limits):
    try:
        values = np.asarray(limits, dtype=float)
    except (TypeError, ValueError) as exc:
        raise LayoutError(f"Invalid position range: {limits!r}") from exc
    if values.shape == (2,):
        values = values.reshape(1, 2)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise LayoutError(f"Expected [low, high] or a list of intervals, got {limits!r}")
    if not np.isfinite(values).all() or np.any(values[:, 0] > values[:, 1]):
        raise LayoutError(f"Position intervals must be finite and ordered: {limits!r}")
    return tuple(tuple(float(v) for v in interval) for interval in values)


def xy_region(xlim, ylim):
    parts = []
    for (x0, x1), (y0, y1) in itertools.product(position_intervals(xlim), position_intervals(ylim)):
        if x0 == x1 and y0 == y1:
            parts.append(Point(x0, y0))
        elif x0 == x1 or y0 == y1:
            parts.append(LineString([(x0, y0), (x1, y1)]))
        else:
            parts.append(box(x0, y0, x1, y1))
    return unary_union(parts)


def unit_quat(quat):
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if quat.shape != (4,) or not np.isfinite(quat).all() or norm == 0:
        raise LayoutError(f"Invalid quaternion: {quat}")
    return quat / norm


class PlacementGenerator(ClutteredGenerator):
    def to_world_pose(self, pose):
        # Parent-to-world precedes object-to-parent, including tilted supports.
        return np.r_[self.frame[:3] + quat2mat(self.frame[3:]) @ pose[:3], unit_quat(qmult(self.frame[3:], pose[3:]))]


class PlacementError(LayoutError):
    """A greedy placement blocked a later mandatory task object."""


class TaskLayoutSampler:
    """Sample the task YAML distribution without importing or launching Isaac.

    Native placement uses numpy's legacy RNG. Its state is restored on exit,
    so sampling does not perturb callers' randomness. Use separate processes
    rather than threads when sampling concurrently.
    """

    def __init__(self, task, *, max_tries=1000):
        self.task = task
        self.max_tries = max_tries
        self.task_cfg = yaml.safe_load((TASK_CONFIGS / f"{task}.yml").read_text())
        defaults = yaml.safe_load((TASK_CONFIGS / "_task.yml").read_text())
        settings = {**defaults.get("common", {}), **defaults.get("tasks", {}).get(task, {})}
        scene = self.task_cfg.get("scene_config", settings.get("scene_config", "default"))
        self.scene_cfg = yaml.safe_load((ROOT / "env_cfg/scene" / f"{scene}.yml").read_text())

    def sample(self, seed):
        previous = np.random.get_state()
        np.random.seed(int(seed))
        try:
            rng = np.random.default_rng(int(seed))
            for retry in range(8):
                try:
                    return self._sample(rng)
                except PlacementError:
                    if retry == 7:
                        raise
        finally:
            np.random.set_state(previous)

    def _sample(self, rng):
        self.sampling_info = {"clutter_groups": []}
        task_cfg = self._equation_config(rng) if self.task == "solve_equation" else self.task_cfg
        layout = {
            key: deepcopy(self.scene_cfg[key]) for key in ("Room", "Table", "Ground", "Light") if key in self.scene_cfg
        }
        for key in ("Room", "Ground", "Light"):
            if key in layout:
                layout[key]["random"] = False
        background = self.scene_cfg.get("Background")
        if background:
            layout["Background"] = {"category_name": background["default"], "intensity": background["intensity"]}
        generators = {}
        if "Table" in layout:
            table = layout["Table"]
            pos, size = np.asarray(table["default_pos"]), np.asarray(table["scale"])
            generators["Table"] = PlacementGenerator(
                frame=[0, 0, pos[2] + size[2] / 2, 1, 0, 0, 0],
                global_container=box(
                    pos[0] - size[0] / 2 + 0.05,
                    pos[1] - size[1] / 2 + 0.05,
                    pos[0] + size[0] / 2 - 0.05,
                    pos[1] + size[1] / 2 - 0.05,
                ),
            )
        generators["Ground"] = PlacementGenerator(-20, -20, 20, 20, frame=[0, 0, GROUND_TOP, 1, 0, 0, 0])
        by_label = {}
        placement_order = ("Geometry", "Dynamic", "Articulation", "Garment", "Rigid", "Fluid")
        pending = [
            (kind, deepcopy(group))
            for cfg in (self.scene_cfg, task_cfg)
            for kind in placement_order
            for group in cfg.get(kind, []) or []
        ]
        while pending:
            for index, (kind, group) in enumerate(pending):
                if self._ready(group, by_label):
                    pending.pop(index)
                    break
            else:
                raise LayoutError(
                    f"Unresolved placement/model dependencies: {[g.get('select_mode') for _, g in pending]}"
                )
            picks = self._picks(kind, group, rng, by_label)
            for slot, (entry, model, label) in enumerate(picks):
                cfg = {
                    **group.get("common", {}),
                    **{k: v for k, v in entry.items() if k not in ("name", "index", "group")},
                }
                dynamic = group.get("dynamic_layout")
                if dynamic:
                    if dynamic["type"] != "x_symmetric_line":
                        raise LayoutError(f"Unsupported dynamic layout: {dynamic}")
                    x = dynamic.get("center", 0) + (slot - (len(picks) - 1) / 2) * dynamic["spacing"]
                    cfg["xlim"] = [x, x]
                label = label or f"{entry['name']}_{slot}"
                if label in by_label:
                    raise LayoutError(f"Duplicate label {label!r}")
                inst = self._place(kind, entry, model, label, cfg, rng, generators, by_label)
                layout.setdefault(kind, {}).setdefault(entry["name"], []).append(inst)
                by_label[label] = (kind, inst)
        self._clutter(layout, generators, rng, by_label)
        return layout

    def _equation_config(self, rng):
        """Published equation layouts contain one hole and an optional sixth pad.

        The task YAML describes the complete template. Its reward initializer
        requires exactly one missing target and at least one available solution.
        Keep the original task/scoring code and produce that saved-layout form.
        """
        operations = ("plus", "minus", "multiplication", "division")
        while True:
            a, b = rng.integers(0, 10, 2).tolist()
            op = str(rng.choice(operations))
            if op == "minus" and a < b or op == "division" and (b == 0 or a % b):
                continue
            result = {
                "plus": lambda: a + b,
                "minus": lambda: a - b,
                "multiplication": lambda: a * b,
                "division": lambda: a // b,
            }[op]()
            break
        two_digits = result >= 10
        missing = int(rng.choice([0, 1, 2, 4, 5] if two_digits else [0, 1, 2, 4]))
        values = {"t0": a, "t2": b, "t4": result // 10 if two_digits else result}
        if two_digits:
            values["t5"] = result % 10
        cfg = deepcopy(self.task_cfg)
        if two_digits:
            pad = deepcopy(next(g for g in cfg["Geometry"] if g["select_mode"]["label"] == ["mat4"]))
            pad["common"]["xlim"] = [0.255, 0.255]
            pad["select_mode"]["label"] = ["mat5"]
            cfg["Geometry"].append(pad)
            digit = deepcopy(next(g for g in cfg["Rigid"] if g["select_mode"].get("label") == ["t4"]))
            digit["common"]["relative_plane"] = "mat5"
            digit["select_mode"]["label"] = ["t5"]
            cfg["Rigid"].append(digit)
        groups = []
        for group in cfg["Rigid"]:
            labels = group["select_mode"].get("label", [])
            label = labels[0] if labels else None
            if label == f"t{missing}":
                continue
            if label in values:
                group["category"] = [{"name": "number", "index": [values[label] + 10 * int(rng.integers(3))]}]
            elif label == "t1":
                group["category"] = [{"name": op, "index": [0]}]
            elif label == "equal":
                group["select_mode"]["label"] = ["t3"]
            groups.append(group)
        # Reserve the fixed operator locations before placing the loose digits.
        cfg["Rigid"] = sorted(groups, key=lambda g: g["select_mode"]["mode"] == "hierarchical")
        self.sampling_info["equation"] = {
            "a": a,
            "b": b,
            "operation": op,
            "result": result,
            "missing_label": f"t{missing}",
            "two_digit_answer": two_digits,
        }
        return cfg

    @staticmethod
    def _ready(group, known):
        mode = group.get("select_mode", {})
        for key in ("same_label", "key_label"):
            if mode.get(key) and mode[key] not in known:
                return False
        if mode.get("linked_label_prefix") and not any(
            label.startswith(mode["linked_label_prefix"] + "_") for label in known
        ):
            return False
        for entry in group["category"]:
            cfg = {**group.get("common", {}), **entry}
            refs = cfg.get("relative_plane", "Table")
            for ref in refs if isinstance(refs, list) else [refs]:
                label = ref.split("/", 1)[0]
                if label.lower() not in ("table", "ground") and label not in known:
                    return False
        return True

    def _picks(self, kind, group, rng, known):
        mode = deepcopy(group.get("select_mode", {}))
        entries = deepcopy(group["category"])
        selection = mode.get("mode", "unique")
        flat_prefix = mode.get("label_prefix") if isinstance(mode.get("label_prefix"), str) else None
        if flat_prefix:
            mode.pop("label_prefix")
        if selection in ("same_as_label", "same_index_as_label", "label_map"):
            _, parent = known[mode.get("same_label", mode.get("key_label"))]
            model = parent["category_idx"]
            if selection == "same_as_label":
                entries = [e for e in entries if e["name"] == parent["category"]]
            elif selection == "label_map":
                mapping = json.loads((OBJECTS / kind / entries[0]["name"] / "map.json").read_text())
                model = int(mapping[f"{parent['category']}/{model:05d}"])
            entries = [
                {**e, "index": [model]} for e in entries if model in e.get("index", available_indices(kind, e["name"]))
            ]
            mode["mode"] = "allow_duplicate"
        elif selection == "linked_count":
            count = sum(label.startswith(mode["linked_label_prefix"] + "_") for label in known)
            mode.update(
                mode="allow_duplicate", nums=count, label=[f"{mode['label_prefix'][0]}_{i}" for i in range(count)]
            )
        if mode.get("allow_global_same") is False:
            used = {(inst["category"], inst["category_idx"]) for k, inst in known.values() if k == kind}
            for entry in entries:
                entry["index"] = [
                    i
                    for i in entry.get("index", available_indices(kind, entry["name"]))
                    if (entry["name"], i) not in used
                ]
        if not entries or not any(e.get("index", available_indices(kind, e["name"])) for e in entries):
            raise LayoutError(f"No models match {selection}: {group.get('select_mode')}")
        picks = _choose_models(entries, mode, rng, kind)
        if flat_prefix:
            picks = [(entry, model, f"{flat_prefix}_{i}") for i, (entry, model, _) in enumerate(picks)]
        return picks

    def _generator(self, reference, generators, known):
        if reference.lower() in ("table", "ground"):
            return generators[reference.capitalize()]
        if reference in generators:
            return generators[reference]
        label, _, tag = reference.partition("/")
        kind, parent = known[label]
        meta = metadata(kind, parent["category"], parent["category_idx"])
        position = np.asarray(parent["default_pos"])
        rotation = unit_quat(parent["default_ori"])
        scale = np.asarray(parent["scale"])
        container = box(-20, -20, 20, 20)
        if tag:
            support = meta.get("passive", {}).get("support", {}).get(tag)
            if not support:
                raise LayoutError(f"Missing support {reference}")
            if support.get("base_link"):
                raise LayoutError(f"Support with articulated base_link needs USD transform: {reference}")
            frames = np.asarray(support["center"], dtype=float).reshape(-1, 7)
            frame = frames[0]
            position = position + quat2mat(rotation) @ (frame[:3] * scale)
            rotation = unit_quat(qmult(rotation, unit_quat(frame[3:])))
            radii = np.broadcast_to(support.get("radius", 20), (len(frames),))
            local_centers = (frames[:, :3] - frame[:3]) @ quat2mat(unit_quat(frame[3:]))
            container = unary_union(
                [
                    Point(center[:2]).buffer(float(radius) * min(scale[:2]))
                    for center, radius in zip(local_centers, radii, strict=True)
                ]
            )
        elif kind == "Dynamic" and parent["category"] == "conveyor":
            # Untagged conveyor offsets are expressed in the world-aligned belt
            # plane at the root XY, rather than in the model's reversed root.
            support = next(iter(meta["passive"]["support"].values()))
            position = position + [0, 0, support["center"][0][2]]
            rotation = np.array([1.0, 0, 0, 0])
        else:
            # An untagged object reference denotes its top plane. Explicit
            # offsets (e.g. fluid z=-.133) then place contents inside it.
            vertices = np.asarray(meta["geometry"]["aligned_bbox"]["vertices"]) * scale
            top = (vertices @ quat2mat(rotation).T)[:, 2].max()
            position = position + [0, 0, top]
        generator = PlacementGenerator(frame=np.r_[position, rotation], global_container=container)
        generators[reference] = generator
        return generator

    def _place(self, kind, entry, model, label, cfg, rng, generators, known, *, clutter=False):
        meta = deepcopy(metadata("Clutter" if clutter else kind, entry["name"], model))
        scale = np.asarray(cfg.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        for bounds in meta.get("geometry", {}).values():
            if isinstance(bounds, dict) and "vertices" in bounds:
                bounds["vertices"] = (np.asarray(bounds["vertices"]) * scale).tolist()
        for place in meta.get("active", {}).get("place", {}).values():
            frame = place.get("projection_circle", {}).get("center")
            if frame:
                place["projection_circle"]["center"] = [*(np.asarray(frame[:3]) * scale), *unit_quat(frame[3:])]
        xlim, ylim = cfg.get("xlim", [0, 0]), cfg.get("ylim", [0, 0])
        region = xy_region(xlim, ylim)
        references = cfg.get("relative_plane", "Table")
        references = list(references) if isinstance(references, list) else [references]
        for ri in rng.permutation(len(references)):
            reference = references[int(ri)]
            generator = self._generator(reference, generators, known)
            options = dict(
                rotate_rand=cfg.get("rotate_rand", False),
                rotate_deg=cfg.get("rotate_deg", 0),
                margin=cfg.get("margin", 0 if kind == "Geometry" else 0.01),
                max_attempts=min(self.max_tries, 50) if clutter else self.max_tries,
                name=label,
                allowed_region=region,
                cluttered=clutter,
                check_mode=cfg.get("check_mode", "bbox"),
            )
            zlim = cfg.get("zlim")
            if kind == "Fluid":
                options["check_mode"] = "enforce"
                zlim = zlim or [0, 0]
            if kind == "Geometry":
                # Task fixtures have explicit positions, including touching
                # baskets and overlapping open frames. Preserve these poses;
                # their footprints still constrain later movable objects.
                options["check_mode"] = "enforce"
            # ProhibitedArea constrains movable objects on the table. Static
            # mats and fixtures may intentionally cover those regions.
            if kind in ("Rigid", "Garment", "Articulation") and reference.lower() == "table":
                areas = self.task_cfg.get("ProhibitedArea", self.task_cfg.get("Prohibited_Area", []))
                prohibited = [box(*area) for area in areas or []]
                temporary = len(generator.placed_polygons)
                for i, area in enumerate(prohibited):
                    generator.add_prohibited_area(area, f"prohibited_{i}")
            else:
                temporary = None
            before = len(generator.placed_polygons)
            ok, pose, _ = generator.add_model_from_config(
                meta,
                place_tag=cfg.get("place_tag"),
                zlim=zlim,
                qpos=unit_quat(cfg.get("qpos", [1, 0, 0, 0])).tolist(),
                **options,
            )
            if ok and cfg.get("check_mode") == "enforce":
                # Enforced poses may be inside open goal frames/containers.
                # They do not occupy the entire bounding-box interior.
                generator.placed_polygons = generator.placed_polygons[:before]
                import rtree.index

                generator.rtree_idx = rtree.index.Index()
                for index, (_, polygon) in enumerate(generator.placed_polygons):
                    generator.rtree_idx.insert(index, polygon.bounds)
            if temporary is not None:
                # Keep successful objects, remove temporary prohibitions.
                placed = generator.placed_polygons[temporary + len(prohibited) :]
                generator.placed_polygons = generator.placed_polygons[:temporary]
                generator.prohibited_area = []
                import rtree.index

                generator.rtree_idx = rtree.index.Index()
                for index, (_, polygon) in enumerate(generator.placed_polygons):
                    generator.rtree_idx.insert(index, polygon.bounds)
                for name, polygon in placed:
                    generator.add_polygon(polygon, name, check_mode="enforce")
            if ok:
                break
        else:
            raise PlacementError(f"Could not place {kind}/{entry['name']}/{model} ({label}) on {references}")
        physics = {**(meta.get("physics") or {}), "type": kind.lower()}
        physics.update(PHYSICS_OVERRIDES.get(self.task, {}).get(entry["name"], {}))
        physics.update(cfg.get("physics", {}) or {})
        return {
            **deepcopy(cfg),
            "category": entry["name"],
            "category_idx": int(model),
            "group": entry.get("group"),
            "xlim": deepcopy(xlim),
            "ylim": deepcopy(ylim),
            "zlim": zlim,
            "qpos": unit_quat(cfg.get("qpos", [1, 0, 0, 0])).tolist(),
            "label": label,
            "relative_plane": reference.lower() if reference.lower() in ("table", "ground") else reference,
            "default_pos": pose[:3].tolist(),
            "default_ori": pose[3:].tolist(),
            "scale": scale.tolist(),
            "need_check_stable": cfg.get("need_check_stable", True),
            "physics": physics,
            "visual": cfg.get("visual", {}) or {},
            **({"type": "cluttered"} if clutter else {}),
        }

    def _clutter(self, layout, generators, rng, known):
        count = 0
        groups = self.task_cfg.get("Clutter", []) or []
        for group_index, group in enumerate(groups):
            used = set()
            catalog = yaml.safe_load((OBJECTS / group["yaml_path"]).read_text())["Clutter"]
            eligible = [cat for cat in catalog if cat not in used]
            if group.get("mode", "category_unique") != "category_unique":
                raise LayoutError(f"Unsupported clutter mode: {group}")
            requested = int(group["nums"])
            for _ in range(requested):
                # Sample the full catalog. A candidate that cannot fit may be
                # replaced, as in rejection sampling; no model whitelist.
                for attempt in range(100):
                    if not eligible:
                        raise LayoutError("Not enough distinct clutter categories")
                    cat = str(rng.choice(eligible))
                    model = int(rng.choice(catalog[cat]))
                    try:
                        inst = self._place(
                            "Rigid",
                            {"name": cat},
                            model,
                            f"clutter_{count}",
                            group,
                            rng,
                            generators,
                            known,
                            clutter=True,
                        )
                    except LayoutError:
                        continue
                    break
                else:
                    # The released eval layouts likewise contain fewer
                    # clutter objects when the configured region fills up.
                    # Task objects remain mandatory; clutter is best effort.
                    break
                inst["clutter_idx"] = group_index
                layout.setdefault("Rigid", {}).setdefault(cat, []).append(inst)
                used.add(cat)
                eligible.remove(cat)
                count += 1
            self.sampling_info["clutter_groups"].append(
                {"index": group_index, "requested": requested, "placed": len(used)}
            )
