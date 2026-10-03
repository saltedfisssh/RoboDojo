"""Cloth mesh transforms are dynamic in PhysX 107.3, as are their local points."""

import numpy as np


def cloth_xform_ops(prim):
    from pxr import UsdGeom

    ops = tuple(UsdGeom.Xformable(prim).GetOrderedXformOps())
    allowed = {UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.TypeScale}
    if not ops or any(op.IsInverseOp() or op.GetOpType() not in allowed for op in ops):
        raise ValueError("Cloth checkpoint requires the live translate/orient/scale transform stack")
    return ops


def capture_xform_ops(ops):
    values = {}
    for index, op in enumerate(ops):
        value = op.Get()
        value = [value.GetReal(), *value.GetImaginary()] if hasattr(value, "GetImaginary") else value
        values[str(index)] = np.asarray(value, dtype=np.float64)
    return values


def restore_xform_ops(ops, values):
    for index, op in enumerate(ops):
        saved, current = values[str(index)], op.Get()
        if hasattr(current, "GetImaginary"):
            value = type(current)(float(saved[0]), type(current.GetImaginary())(*saved[1:]))
        else:
            value = type(current)(*saved)
        op.Set(value)


def world_particle_positions(prim, local_positions):
    from pxr import Usd, UsdGeom

    # Gf matrices use row vectors; translation occupies the last row.
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return np.asarray(local_positions) @ transform[:3, :3] + transform[3, :3]
