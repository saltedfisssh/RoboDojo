from itertools import product

import numpy as np
import pytest
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from src.datagen.collision_world import world_box_in_robot_frame


def test_world_obstacle_corners_survive_rotated_robot_frame_and_bias():
    lower, upper = np.array([-0.02, -0.48, 0.72]), np.array([0.02, -0.45, 1.30])
    pose = np.r_[[0.3, -0.5, 0.2], euler2quat(0.2, -0.1, 1.3)]
    bias = np.array([0.03, -0.02, 0.01])
    box = world_box_in_robot_frame(lower, upper, pose, bias, padding=0)
    corners = np.array(list(product((-1, 1), repeat=3))) * np.array(box["dims"]) / 2
    in_robot = corners @ quat2mat(box["pose"][3:]).T + box["pose"][:3]
    in_world = (in_robot - bias) @ quat2mat(pose[3:]).T + pose[:3]
    np.testing.assert_allclose(in_world.min(axis=0), lower, atol=1e-14)
    np.testing.assert_allclose(in_world.max(axis=0), upper, atol=1e-14)


def test_static_obstacles_fail_closed_on_invalid_bounds():
    for lower, upper in [([0, 0, 0], [1, 0, 1]), ([0, 0, np.nan], [1, 1, 1])]:
        with pytest.raises(ValueError):
            world_box_in_robot_frame(lower, upper, [0, 0, 0, 1, 0, 0, 0], [0, 0, 0])
