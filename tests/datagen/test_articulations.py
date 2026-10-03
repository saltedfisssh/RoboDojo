"""Exercise serialized controller actions against Sim 5.1's backend contract."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from src.datagen.articulations import restore


@pytest.mark.parametrize("sleeping", [None, False, True])
def test_restore_converts_serialized_joint_indices_for_torch_backend(monkeypatch, sleeping):
    types = ModuleType("isaacsim.core.utils.types")
    types.ArticulationAction = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.types", types)
    pxr = ModuleType("pxr")
    pxr.UsdPhysics = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    values = {}

    class Articulation:
        dof_names = ["button"]

        def get_joint_positions(self):
            return torch.tensor([0.002], dtype=torch.float32)

        get_joint_velocities = get_joint_positions
        get_applied_joint_efforts = get_joint_positions

        def set_joint_positions(self, value):
            assert isinstance(value, torch.Tensor) and value.dtype == torch.float32
            values["position"] = value

        def set_joint_velocities(self, value):
            values["velocity"] = value

        def set_joint_efforts(self, value):
            values["effort"] = value

        def apply_action(self, action):
            # Sim 5.1 torch.resolve_indices only converts Python lists; a
            # serialized NumPy array otherwise reaches .to() and crashes.
            indices = action.joint_indices
            if isinstance(indices, list):
                indices = torch.tensor(indices, dtype=torch.long)
            values["indices"] = indices.to(dtype=torch.long)
            values["target"] = action.joint_positions

    arrays = {
        "joint_positions": np.array([0.001]),
        "joint_velocities": np.array([0.003]),
        "joint_efforts": np.array([0.004]),
        "action_joint_positions": np.array([0.006]),
        "action_joint_indices": np.array([0], dtype=np.int64),
    }

    def restore_sleep_after_targets(obj, saved_sleeping):
        assert "target" in values and "position" in values and "velocity" in values
        values["sleeping"] = saved_sleeping

    monkeypatch.setattr("src.datagen.articulations.restore_sleep", restore_sleep_after_targets)
    context = {"dof_names": ["button"], "drives": []}
    if sleeping is not None:
        context["sleeping"] = sleeping
    restore(Articulation(), arrays, context)
    if sleeping is None:
        assert "sleeping" not in values  # Legacy checkpoints retain their existing behavior.
    else:
        assert values["sleeping"] is sleeping
    np.testing.assert_allclose(values["position"].numpy(), [0.001])
    np.testing.assert_allclose(values["target"], [0.006])
    assert values["indices"].tolist() == [0]


def test_sibling_joint_paths_stay_inside_imported_asset():
    pxr = pytest.importorskip("pxr.Usd")
    from src.datagen.articulations import asset_prim, resolve_drive_path

    stage = pxr.Stage.CreateInMemory()
    root = stage.DefinePrim("/World/laptop", "Xform")
    body = stage.DefinePrim("/World/laptop/body", "Xform")
    stage.DefinePrim("/World/laptop/screen/hinge", "PhysicsRevoluteJoint")
    obj = SimpleNamespace(stage=stage, prim=body, usd_prim_path=str(root.GetPath()))
    assert asset_prim(obj) == root
    expected = "/World/laptop/screen/hinge"
    assert str(resolve_drive_path(obj, "screen/hinge", "asset")) == expected
    assert str(resolve_drive_path(obj, "../screen/hinge", "view")) == expected
    with pytest.raises(ValueError, match="escapes"):
        resolve_drive_path(obj, "../../other/hinge", "view")
    with pytest.raises(ValueError, match="relative"):
        resolve_drive_path(obj, expected, "asset")
