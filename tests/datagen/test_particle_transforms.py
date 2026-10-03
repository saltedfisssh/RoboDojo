"""A cloth's identical local points can represent different physical world states."""

import numpy as np
import pytest

pxr = pytest.importorskip("pxr.Usd")
from pxr import Gf, UsdGeom  # noqa: E402

from src.datagen.particle_transforms import (  # noqa: E402
    capture_xform_ops,
    cloth_xform_ops,
    restore_xform_ops,
    world_particle_positions,
)


def test_restoring_local_particles_alone_does_not_restore_cloth_world_state():
    stage = pxr.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/world")
    parent.AddTranslateOp().Set(Gf.Vec3d(0.2, -0.1, 0.0))
    mesh = UsdGeom.Mesh.Define(stage, "/world/cloth")
    translation = mesh.AddTranslateOp()
    translation.Set(Gf.Vec3d(0.0, 0.0, 0.8))
    rotation = mesh.AddOrientOp()
    rotation.Set(Gf.Quatf(1, 0, 0, 0))
    mesh.AddScaleOp().Set(Gf.Vec3f(1))
    points = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], dtype=np.float32)
    mesh.GetPointsAttr().Set(points)
    ops = cloth_xform_ops(mesh.GetPrim())
    saved = capture_xform_ops(ops)
    expected = world_particle_positions(mesh.GetPrim(), points)
    np.testing.assert_allclose(expected[0], [0.2, -0.1, 0.8])
    translation.Set(Gf.Vec3d(0.0, 0.0, 0.9))
    rotation.Set(Gf.Quatf(np.sqrt(0.5), 0, 0, np.sqrt(0.5)))
    # The old schema's particle-only readback is exactly equal despite a
    # 100 mm physical translation and 90-degree orientation change.
    np.testing.assert_array_equal(np.asarray(mesh.GetPointsAttr().Get()), points)
    assert np.max(np.linalg.norm(world_particle_positions(mesh.GetPrim(), points) - expected, axis=1)) > 0.1
    restore_xform_ops(ops, saved)
    np.testing.assert_allclose(world_particle_positions(mesh.GetPrim(), points), expected, atol=1e-7)


def test_unhandled_transform_stacks_are_rejected_instead_of_losing_state():
    stage = pxr.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/cloth")
    mesh.AddTransformOp().Set(Gf.Matrix4d(1))
    with pytest.raises(ValueError, match="transform stack"):
        cloth_xform_ops(mesh.GetPrim())
