"""Keep render-only Kit updates from changing a constant conveyor's physical state."""

from functools import wraps


def preserve_attributes_during_render(render, attributes):
    """Restore the live values, including checkpoint-restored speeds, after rendering."""
    attributes = tuple(attributes)

    @wraps(render)
    def guarded(*args, **kwargs):
        values = [attribute.Get() for attribute in attributes]
        try:
            return render(*args, **kwargs)
        finally:
            for attribute, value in zip(attributes, values, strict=True):
                if attribute.Get() != value:
                    attribute.Set(value)

    return guarded


def guard_constant_conveyor_render(env):
    """Only the shipped conveyor is admitted by this data-generation adapter.

    Sim 5.1 SimulationContext.render disables /app/player/playSimulations
    during its Kit update. That update runs the conveyor graph and was observed
    to zero PhysxSurfaceVelocity despite its unchanged -0.1 m/s graph input.
    A camera capture must preserve the pre-render physical velocity.
    """
    from pxr import Usd

    lm = env.scene_manager.layout_manager
    attributes = []
    for record in lm.get_layout_records(0, "Dynamic"):
        if record["category"] != "conveyor" or record["category_idx"] != 0:
            raise ValueError("Render guard only supports the shipped constant-speed conveyor")
        obj = lm.get_scene_object(0, record["inst_name"])
        root = obj.stage.GetPrimAtPath(obj.prim_path)
        for prim in Usd.PrimRange(root):
            for name in ("physxSurfaceVelocity:surfaceVelocity", "physxSurfaceVelocity:surfaceAngularVelocity"):
                attribute = prim.GetAttribute(name)
                if attribute and attribute.HasAuthoredValueOpinion():
                    attributes.append(attribute)
    if not attributes:
        raise ValueError("Conveyor render guard found no authored surface velocities")
    env.render = preserve_attributes_during_render(env.render, attributes)
    print("[datagen] Camera renders preserve constant conveyor surface velocities.", flush=True)
