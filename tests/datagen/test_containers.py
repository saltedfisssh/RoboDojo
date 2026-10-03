import numpy as np
from transforms3d.euler import euler2mat, euler2quat
from transforms3d.quaternions import quat2mat

from src.datagen.programs.containers import upright_orientation


def test_rolled_pen_is_stood_up_without_swinging_the_wrist_behind_the_holder():
    # The real failed grasp rolled the pen about its long X axis by ~74 deg.
    # Fixing its full orientation to Ry(-90) unnecessarily swings the wrist
    # around the holder. Axial roll is irrelevant to tip-down insertion.
    held_q = euler2quat(np.deg2rad(74), 0, 0)
    target = quat2mat(upright_orientation(held_q, [1, 0, 0]))
    np.testing.assert_allclose(target @ [1, 0, 0], [0, 0, 1], atol=1e-8)
    delta = target @ quat2mat(held_q).T
    tool = delta @ euler2mat(0, np.pi / 2, 0)
    np.testing.assert_allclose(tool[:, 0], [1, 0, 0], atol=1e-8)


def test_marker_uses_its_local_z_axis_and_preserves_an_already_upright_grasp():
    held_q = euler2quat(0, 0, 0.7)
    np.testing.assert_allclose(quat2mat(upright_orientation(held_q, [0, 0, 1])), quat2mat(held_q), atol=1e-8)
