"""Scoped scene fixes for malformed assets; original USD files remain immutable."""

from contextlib import contextmanager

ADAPTERS = {
    "pour_liquid_into_cup": {"wuliangye_single_rigid_body_v2": "wuliangye"},
    "pour_balls_into_vase": {"sphere_single_rigid_body_v1": "sphere"},
}


def selected_adapters(task, saved_changes=None):
    """New scenes use current fixes; replay uses exactly the recorded versions."""
    available = ADAPTERS.get(task, {})
    if saved_changes is None:
        return dict(available)
    names = {change["adapter"] for change in saved_changes}
    unknown = names - available.keys()
    if unknown:
        raise ValueError(f"Unsupported recorded scene adaptations for {task}: {sorted(unknown)}")
    return {name: category for name, category in available.items() if name in names}


@contextmanager
def single_body_assets(task, saved_changes=None):
    """Compose malformed rigid assets under their RoboDojo root rigid body.

    The shipped wuliangye and sphere assets author RigidBodyAPI on descendants;
    RoboDojo's RigidObject also authors it on the root. PhysX rejects that nested
    enabled-body hierarchy. Remove the descendant body API in this scene only,
    preserving its collider, material and geometry. Articulations are excluded.
    """
    changes = []
    selected = selected_adapters(task, saved_changes)
    if not selected:
        yield changes
        return
    from pxr import Usd, UsdPhysics

    from env.scene_manager.objects import rigid

    original = rigid.add_reference_to_stage

    def add_reference(usd_path, prim_path, **kwargs):
        root = original(usd_path=usd_path, prim_path=prim_path, **kwargs)
        adapter = next((name for name, category in selected.items() if f"/{category}/" in str(prim_path)), None)
        if adapter is not None:
            # Do this BEFORE SingleRigidPrim initializes its root tensor view.
            # Removing the nested API later in _setup_physics is too late:
            # PhysX has already registered an invalid body hierarchy by then.
            descendants = list(Usd.PrimRange(root))[1:]
            if any(prim.IsA(UsdPhysics.Joint) for prim in descendants):
                raise ValueError("The asset contains joints and cannot be composed as one rigid body")
            for prim in descendants:
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    changes.append(
                        {
                            "adapter": adapter,
                            "prim": str(prim.GetPath()),
                            "relative_prim": str(prim.GetPath().MakeRelativePath(root.GetPath())),
                            "removed_api": "PhysicsRigidBodyAPI",
                        }
                    )
        return root

    rigid.add_reference_to_stage = add_reference
    try:
        yield changes
    finally:
        rigid.add_reference_to_stage = original


def adaptation_signature(env):
    """Stable scene identity independent of generated instance IDs or reloads."""
    return [
        list(item)
        for item in sorted(
            {
                (change["adapter"], change["relative_prim"], change["removed_api"])
                for change in getattr(env, "datagen_scene_adaptations", [])
            }
        )
    ]
