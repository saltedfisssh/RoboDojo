"""Regression checks for the lid mismatch gate used during checkpoint rendering."""

import numpy as np
import pytest
from transforms3d.euler import euler2mat
from transforms3d.quaternions import axangle2quat, quat2mat

from src.datagen.visual_state import local_pose_from_world, pose_errors, pose_from_usd_matrix


def test_detects_open_lid_even_when_positions_and_quaternion_norms_match():
    closed = [[0, 0, 0.8, 1, 0, 0, 0]]
    open_lid = [[0, 0, 0.8, np.sqrt(0.5), -np.sqrt(0.5), 0, 0]]
    position, angle = pose_errors(open_lid, closed)
    np.testing.assert_allclose(position, 0)
    np.testing.assert_allclose(angle, np.pi / 2)


def test_sign_and_float_norm_noise_do_not_report_false_visual_motion():
    physics = np.array([[0, 0, 0.8, 1.0000004, 0, 0, 0]])
    usd = np.array([[0.003, 0.004, 0.8, -1, 0, 0, 0]])
    position, angle = pose_errors(usd, physics)
    np.testing.assert_allclose(position, 0.005)
    np.testing.assert_allclose(angle, 0)


@pytest.mark.parametrize("quaternion", [[0, 0, 0, 0], [np.nan, 0, 0, 1]])
def test_invalid_visual_pose_is_rejected(quaternion):
    with pytest.raises(ValueError):
        pose_errors([[0, 0, 0, *quaternion]], [[0, 0, 0, 1, 0, 0, 0]])


@pytest.mark.parametrize("angle", [0.0001352431645, np.pi - 1e-7, -np.pi / 2])
@pytest.mark.parametrize("scale", [0.01, 1.0, 2.5])
def test_small_and_near_pi_world_rotation_survive_scaled_rotated_parent(angle, scale):
    parent = np.eye(4)
    parent[:3, :3] = scale * euler2mat(0.3, -0.8, 1.1).T
    parent[3, :3] = [0.1, -0.2, 0.7]
    expected = np.r_[[0, -0.15988344, 0.81334108], axangle2quat([1, 0, 0], angle)]
    local = local_pose_from_world(parent, expected)
    authored = np.eye(4)
    # Isaac's USD link orientation attributes can be quatf. Retain that
    # precision in the round trip while preserving the original scale.
    authored[:3, :3] = quat2mat(local[3:].astype(np.float32)).T / scale
    authored[3, :3] = local[:3]
    distance, error = pose_errors(pose_from_usd_matrix(authored @ parent), expected)
    assert distance < 1e-12
    assert error < 2e-7


def test_near_identity_rotation_is_not_lost_when_diagonal_rounds_to_one():
    expected = [0, 0, 0.8, 1, -6.762157863704488e-5, 0, 0]
    matrix = np.eye(4)
    matrix[:3, :3] = quat2mat(expected[3:]).T.astype(np.float32)
    matrix[3, :3] = expected[:3]
    _, error = pose_errors(pose_from_usd_matrix(matrix), expected)
    assert error < 1e-10


@pytest.mark.parametrize("angle", [0.00016882884, 0.000032])
def test_real_usd_float_orientation_preserves_small_rotation_after_promotion(angle):
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom

    from src.datagen.visual_state import ensure_double_orientation

    stage = Usd.Stage.CreateInMemory()
    xform = UsdGeom.Xform.Define(stage, "/link")
    orient = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    quaternion = axangle2quat([0, 1, 0], angle)
    orient.Set(Gf.Quatf(*quaternion))
    before = pose_from_usd_matrix(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    expected = np.r_[[0, 0, 0], quaternion]
    # Exercise the installed USD conversion, which a numpy-only transform
    # round trip does not reproduce for near-identity quatf attributes.
    assert pose_errors(before, expected)[1] > angle * 0.9
    order = xform.GetXformOpOrderAttr().Get()
    assert ensure_double_orientation(xform)
    assert orient.GetAttr().GetTypeName() == Sdf.ValueTypeNames.Quatd
    assert xform.GetXformOpOrderAttr().Get() == order
    after = pose_from_usd_matrix(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    assert pose_errors(after, expected)[1] < 1e-9
    assert not ensure_double_orientation(xform)
