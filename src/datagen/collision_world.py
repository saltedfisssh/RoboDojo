"""Opt-in static scene bounds for the installed cuRobo dev100/v2 planners."""

from copy import deepcopy
import json

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat


def world_box_in_robot_frame(lower, upper, robot_pose, frame_bias, padding=0.002):
    """Keep a world AABB's orientation when expressing it in the planner frame."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    pose, bias = np.asarray(robot_pose, float), np.asarray(frame_bias, float)
    if lower.shape != (3,) or upper.shape != (3,) or pose.shape != (7,) or bias.shape != (3,):
        raise ValueError("Invalid obstacle or robot pose dimensions")
    if not np.isfinite(np.r_[lower, upper, pose, bias, padding]).all() or np.any(upper <= lower) or padding < 0:
        raise ValueError("Invalid static obstacle bounds")
    if np.linalg.norm(pose[3:]) < 1e-8:
        raise ValueError("Invalid robot orientation")
    inverse = quat2mat(pose[3:]).T
    center = inverse @ ((lower + upper) / 2 - pose[:3]) + bias
    return {"dims": (upper - lower + 2 * padding).tolist(), "pose": np.r_[center, mat2quat(inverse)].tolist()}


def include_static_geometry(p, label="camera_stand"):
    """Add each live USD mesh's world bounds, preserving the existing table.

    This conservatively covers the named static geometry only. Held tools,
    other robots and task objects are not represented by this helper.
    """
    from curobo.scene import Scene
    from pxr import Usd, UsdGeom

    from src.datagen.articulations import asset_prim

    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
    root = asset_prim(obj)
    boxes = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        if not len(points):
            continue
        transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world = points @ transform[:3, :3] + transform[3, :3]
        lower, upper = world.min(axis=0), world.max(axis=0)
        # Planar components still need a positive cuboid extent.
        half = np.maximum((upper - lower) / 2, 1e-5)
        center = (lower + upper) / 2
        boxes.append({"prim": str(prim.GetPath()), "min": (center - half).tolist(), "max": (center + half).tolist()})
    if not boxes:
        raise ValueError(f"No mesh bounds found for static obstacle {label}")
    updates = []
    for robot in p.actions.robots:
        solver = p.rm.planner[robot.robot_name]
        model = deepcopy(solver.scene_model)
        for index, box in enumerate(boxes):
            model.setdefault("cuboid", {})[f"datagen_{label}_{index}"] = world_box_in_robot_frame(
                box["min"], box["max"], robot.entity_origin_pose, solver.frame_bias
            )
        consumers = (solver.motion_planner, solver.motion_planner_batch, solver.ik_solver)
        for consumer in consumers:
            data = consumer.scene_collision_checker.data.cuboids
            if data is None or data.max_n < len(model["cuboid"]):
                raise ValueError("cuRobo static obstacle update exceeds its preallocated cuboid cache")
        updates.append((solver, model, consumers))
    # Validate every cache before mutating any of the three independent worlds.
    for solver, model, consumers in updates:
        scene = Scene.create(model)
        for consumer in consumers:
            consumer.update_world(scene)
        solver.scene_model = model
    p.recorder.path.with_suffix(".planning_obstacles.json").write_text(
        json.dumps(
            {
                "scope": "Static mesh AABBs added to motion, batch motion and IK worlds",
                "boxes": boxes,
                "padding_m": 0.002,
            },
            indent=2,
        )
        + "\n"
    )
    return boxes
