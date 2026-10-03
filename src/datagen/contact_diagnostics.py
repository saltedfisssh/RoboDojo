"""Opt-in contact observations for checkpoint probes in Isaac Sim 5.1.

API reference: the installed PhysX 107.3.26 ContactReportDemo and
PhysxContactReportAPI.test_contact_report_found_lost_runtime_add_remove.
This adds reporting only; it does not change physical contact parameters.
"""

from contextlib import contextmanager
import json

import numpy as np


@contextmanager
def contact_report(planner, *, exclude_articulations=()):
    import carb
    from omni.physx import get_physx_simulation_interface
    import omni.usd
    from pxr import PhysicsSchemaTools, PhysxSchema, UsdPhysics

    from src.datagen.articulations import asset_prim

    # Lab 2.3.2 disables report processing during SimulationContext setup.
    # Its ContactSensor explicitly re-enables this same observer switch.
    settings = carb.settings.get_settings()
    processing_key = "/physics/disableContactProcessing"
    previous_processing = settings.get_as_bool(processing_key)
    settings.set_bool(processing_key, False)
    stage = omni.usd.get_context().get_stage()
    lm = planner.env.scene_manager.layout_manager
    excluded = [
        asset_prim(lm.get_scene_object(0, lm.get_instance_name(0, label))).GetPath() for label in exclude_articulations
    ]
    changed = []
    pairs = {}
    errors = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if any(prim.GetPath().HasPrefix(path) for path in excluded):
            # Reporting from the other actor still observes robot contacts.
            # Changing some instanced assets' API schemas rebuilds collision
            # prims and invalidates existing tensor views.
            continue
        existed = prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        attr = api.GetThresholdAttr()
        authored = attr.HasAuthoredValueOpinion()
        value = attr.Get()
        api.CreateThresholdAttr().Set(0.0)
        changed.append((prim, existed, authored, value))

    def observe(headers, data):
        try:
            for header in headers:
                if not header.num_contact_data:
                    continue
                colliders = tuple(
                    str(PhysicsSchemaTools.intToSdfPath(value)) for value in (header.collider0, header.collider1)
                )
                record = pairs.setdefault(
                    colliders,
                    {
                        "contacts": 0,
                        "max_impulse": 0.0,
                        "min_separation_m": float("inf"),
                        "first_observed_frame": planner.recorder.count,
                    },
                )
                start = header.contact_data_offset
                for contact in data[start : start + header.num_contact_data]:
                    magnitude = float(np.linalg.norm(np.asarray(contact.impulse)))
                    record["contacts"] += 1
                    record["last_observed_frame"] = planner.recorder.count
                    if magnitude > 0:
                        record.setdefault("first_positive_impulse_frame", planner.recorder.count)
                    if magnitude > 0 or float(contact.separation) <= 0:
                        record.setdefault("first_force_or_penetration_frame", planner.recorder.count)
                    record["min_separation_m"] = min(record["min_separation_m"], float(contact.separation))
                    if magnitude >= record["max_impulse"]:
                        record.update(
                            max_impulse=magnitude,
                            position=list(contact.position),
                            normal=list(contact.normal),
                            frame=planner.recorder.count,
                        )
        except Exception as exc:
            if len(errors) < 5:
                errors.append(f"{type(exc).__name__}: {exc}")

    subscription = get_physx_simulation_interface().subscribe_contact_report_events(observe)
    try:
        yield
    finally:
        subscription = None
        settings.set_bool(processing_key, previous_processing)
        for prim, existed, authored, value in changed:
            if not existed:
                prim.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
            else:
                attr = PhysxSchema.PhysxContactReportAPI(prim).GetThresholdAttr()
                attr.Set(value) if authored else attr.Clear()
        result = {
            "scope": "PhysX contact observations during a diagnostic suffix; no task-success claim",
            "reporting_bodies": len(changed),
            "excluded_articulations": list(exclude_articulations),
            "contact_processing_was_disabled": previous_processing,
            "contacts_observed": bool(pairs),
            "callback_errors": errors,
            "pairs": sorted(
                ({"colliders": list(key), **value} for key, value in pairs.items()),
                key=lambda row: -row["max_impulse"],
            ),
        }
        planner.recorder.path.with_suffix(".contacts.json").write_text(json.dumps(result, indent=2) + "\n")
        # Retain the subscription throughout the physics steps above.
        del subscription
