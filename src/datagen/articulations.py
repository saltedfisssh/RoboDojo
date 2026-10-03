"""External SingleArticulation checkpoints using Isaac Sim 5.1 public APIs."""

import numpy as np
import torch


def backend_like(value, reference):
    if isinstance(reference, torch.Tensor):
        return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return np.asarray(value, dtype=np.asarray(reference).dtype)


def asset_prim(obj):
    """Return the imported asset root, which can contain sibling rigid links.

    Sim 5.1 SingleArticulation.prim exposes its selected articulation root link.
    RoboDojo keeps the original reference path separately in usd_prim_path.
    """
    path = getattr(obj, "usd_prim_path", None)
    prim = obj.stage.GetPrimAtPath(path) if path else obj.prim
    if not prim or not prim.IsValid():
        raise ValueError(f"Missing articulation asset root: {path}")
    return prim


def resolve_drive_path(obj, relative_path, base_kind):
    from pxr import Sdf

    root = asset_prim(obj).GetPath()
    relative = Sdf.Path(relative_path)
    if relative.IsAbsolutePath():
        raise ValueError("Drive path must be relative to the articulation")
    if base_kind not in ("asset", "view"):
        raise ValueError(f"Unknown drive path base: {base_kind}")
    base = root if base_kind == "asset" else obj.prim.GetPath()
    path = relative.MakeAbsolutePath(base)
    if not path.HasPrefix(root):
        raise ValueError("Drive path escapes the articulation asset")
    return path


def initialize_authored_drives(env, layout):
    """Set explicitly opted-in fixture drives before settling the initial scene.

    SingleArticulation.apply_action is the Sim 5.1 controller API. This changes
    actuator targets, not joint positions; the fixture settles through physics.
    The opt-in lives in the saved layout, so restored scenes use the same setup.
    """
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdGeom, UsdPhysics

    lm = env.scene_manager.layout_manager
    for records in layout.get("Articulation", {}).values():
        for record in records:
            if not record.get("physics", {}).get("initialize_from_authored_drive_targets"):
                continue
            obj = lm.get_scene_object(0, lm.get_instance_name(0, record["label"]))
            indices, positions = [], []
            for prim, drive_type in obj._iter_drive_joint_prims():
                if prim.GetName() not in obj.dof_names:
                    continue
                target = UsdPhysics.DriveAPI.Get(prim, drive_type).GetTargetPositionAttr().Get()
                if target is None:
                    raise ValueError(f"No authored drive target for {prim.GetPath()}")
                if drive_type == "angular":
                    target = np.deg2rad(target)
                else:
                    target *= UsdGeom.GetStageMetersPerUnit(obj.stage)
                indices.append(obj.dof_names.index(prim.GetName()))
                positions.append(target)
            if not indices:
                raise ValueError(f"No named fixture drives for {record['label']}")
            obj.apply_action(ArticulationAction(joint_positions=np.asarray(positions), joint_indices=indices))
            print(f"[datagen] Initial authored drive targets: {record['label']} {positions}", flush=True)


def capture(obj):
    from pxr import UsdPhysics

    arrays = {
        "joint_positions": np.asarray(obj.get_joint_positions()),
        "joint_velocities": np.asarray(obj.get_joint_velocities()),
        "joint_efforts": np.asarray(obj.get_applied_joint_efforts()),
    }
    action = obj.get_applied_action()
    if action is not None:
        for field in ("joint_positions", "joint_velocities", "joint_efforts", "joint_indices"):
            value = getattr(action, field, None)
            if value is not None:
                arrays[f"action_{field}"] = np.asarray(value)
    drives = []
    for prim, drive_type in obj._iter_drive_joint_prims():
        drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
        drives.append(
            {
                "relative_path": str(prim.GetPath().MakeRelativePath(asset_prim(obj).GetPath())),
                "type": drive_type,
                "position": drive.GetTargetPositionAttr().Get(),
                "velocity": drive.GetTargetVelocityAttr().Get(),
            }
        )
    return arrays, {
        "dof_names": list(obj.dof_names),
        "drives": drives,
        "drive_path_base": "asset",
        "sleeping": is_sleeping(obj),
    }


def restore(obj, arrays, context):
    from isaacsim.core.utils.types import ArticulationAction
    from pxr import UsdPhysics

    if list(obj.dof_names) != context["dof_names"]:
        raise ValueError("External articulation joint names changed")
    for item in context["drives"]:
        # Older checkpoints used the selected root link as their base. A
        # sibling screen joint is valid if it stays within the imported asset.
        path = resolve_drive_path(obj, item["relative_path"], context.get("drive_path_base", "view"))
        drive = UsdPhysics.DriveAPI.Get(obj.stage.GetPrimAtPath(path), item["type"])
        if not drive:
            raise ValueError(f"Missing articulation drive {path}")
        if item["position"] is not None:
            drive.GetTargetPositionAttr().Set(item["position"])
        if item["velocity"] is not None:
            drive.GetTargetVelocityAttr().Set(item["velocity"])
    for field, getter, setter in (
        ("joint_positions", obj.get_joint_positions, obj.set_joint_positions),
        ("joint_velocities", obj.get_joint_velocities, obj.set_joint_velocities),
        ("joint_efforts", obj.get_applied_joint_efforts, obj.set_joint_efforts),
    ):
        setter(backend_like(arrays[field], getter()))
    action = {key.removeprefix("action_"): value for key, value in arrays.items() if key.startswith("action_")}
    if action:
        # Sim 5.1 converts action values for its backend, but torch
        # resolve_indices accepts a list or tensor, not a NumPy array.
        if "joint_indices" in action:
            action["joint_indices"] = np.asarray(action["joint_indices"], dtype=np.int64).tolist()
        obj.apply_action(ArticulationAction(**action))
    if "sleeping" in context:
        # Pose, velocity and target setters can wake an articulation. Restore
        # its saved sleep state only after all these setters have finished.
        restore_sleep(obj, context["sleeping"])


def sleep_handle(obj):
    """PhysX 107.3.26 uses a stage ID and an encoded root/body USD path."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools, UsdUtils

    stage_id = UsdUtils.StageCache.Get().GetId(obj.stage).ToLongInt()
    path = PhysicsSchemaTools.sdfPathToInt(obj.prim.GetPath())
    return get_physx_simulation_interface(), stage_id, path


def is_sleeping(obj):
    interface, stage_id, path = sleep_handle(obj)
    return bool(interface.is_sleeping(stage_id, path))


def restore_sleep(obj, sleeping):
    interface, stage_id, path = sleep_handle(obj)
    method = interface.put_to_sleep if sleeping else interface.wake_up
    method(stage_id, path)


def diagnostics(obj):
    """Compare live tensor and USD root state with PhysX's sleeping flag."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdUtils

    stage_id = UsdUtils.StageCache.Get().GetId(obj.stage).ToLongInt()
    path = obj.prim.GetPath()
    transform = UsdGeom.Xformable(obj.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    quaternion = transform.ExtractRotationQuat()
    pos, quat = obj.get_world_pose()
    return {
        "view_root": str(path),
        "asset_root": str(asset_prim(obj).GetPath()),
        "sleeping": bool(get_physx_simulation_interface().is_sleeping(stage_id, PhysicsSchemaTools.sdfPathToInt(path))),
        "tensor_root_pose": np.r_[pos, quat].tolist(),
        "tensor_root_velocity": np.r_[obj.get_linear_velocity(), obj.get_angular_velocity()].tolist(),
        "usd_root_pose": [*transform.ExtractTranslation(), quaternion.GetReal(), *quaternion.GetImaginary()],
        "usd_velocity": list(obj.prim.GetAttribute("physics:velocity").Get() or []),
        "usd_angular_velocity_deg_s": list(obj.prim.GetAttribute("physics:angularVelocity").Get() or []),
    }
