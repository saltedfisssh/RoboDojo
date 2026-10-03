import numpy as np

from src.datagen.ik import approach_seed_configs, cartesian_waypoints, local_seed_configs, nearest_valid_solution


def test_contact_ik_uses_success_mask_and_named_active_joints():
    # The closest solution failed collision checks. The cost-ranked first
    # valid solution changes branch. Locked fingers are not arm joint deltas.
    positions = np.array([[[1.0, 0.0, 0.9], [0.01, 0.0, 0.01], [0.1, 9.0, 0.2]]])
    chosen = nearest_valid_solution(positions, [[True, False, True]], ["b", "finger", "a"], ["a", "b"], [0, 0])
    np.testing.assert_array_equal(chosen, [0.2, 0.1])


def test_contact_ik_rejects_nonfinite_and_full_turn_jumps():
    assert nearest_valid_solution([[[np.nan], [2 * np.pi], [0.01]]], [[True, True, False]], ["a"], ["a"], [0]) is None


def test_local_contact_seeds_preserve_winding_and_joint_limits():
    current = np.array([2 * np.pi + 0.1, -3.13, 0.01])
    limits = np.array([[-10, -3.14, 0], [10, 3.14, 0.044]])
    seeds = local_seed_configs(current, limits, 32)
    assert seeds.shape == (32, 3)
    np.testing.assert_array_equal(seeds[0], current)
    assert np.all(seeds >= limits[0]) and np.all(seeds <= limits[1])
    assert np.max(np.abs(seeds - current)) <= 0.12 + 1e-12
    # Alternative winding remains forbidden even after locally seeding IK.
    assert (
        nearest_valid_solution(seeds - [2 * np.pi, 0, 0], np.ones(32), ["a", "b", "c"], ["a", "b", "c"], current)
        is None
    )


def test_in_place_rotation_uses_bounded_shortest_arc_steps():
    start = np.array([0, 0, 1, 1, 0, 0, 0])
    # Negative quaternion represents the same +90 degree Y rotation.
    target_q = -np.array([np.sqrt(0.5), 0, np.sqrt(0.5), 0])
    path = np.array(list(cartesian_waypoints(start, start[:3], target_q, 0.004)))
    assert len(path) >= 30
    np.testing.assert_allclose(path[:, :3], np.broadcast_to(start[:3], path[:, :3].shape))
    quats = np.vstack([start[3:], path[:, 3:]])
    angles = 2 * np.arccos(np.clip(np.sum(quats[1:] * quats[:-1], axis=1), -1, 1))
    assert angles.max() <= np.deg2rad(3) + 1e-7
    np.testing.assert_allclose(path[-1, 3:], -target_q, atol=1e-8)


def test_approach_search_keeps_actual_winding_and_respects_offset_limits():
    current = [8.0, -3.13, 0.02, 4.5]
    limits = np.array([[-10, -3.14, 0.01, 4.0], [10, 3.14, 0.044, 5.0]])
    seeds = approach_seed_configs(current, limits, 32)
    np.testing.assert_array_equal(seeds[0], current)
    assert np.all(seeds >= limits[0]) and np.all(seeds <= limits[1])
    assert np.all(np.abs(seeds[2:, :2]) <= np.pi)
