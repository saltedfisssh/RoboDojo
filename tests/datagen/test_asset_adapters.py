import pytest

from src.datagen.asset_adapters import selected_adapters


def test_scene_fixes_are_pinned_to_recorded_episode_versions():
    task = "pour_balls_into_vase"
    assert selected_adapters(task) == {"sphere_single_rigid_body_v1": "sphere"}
    # Loading a historical checkpoint must not silently change its physics.
    assert selected_adapters(task, []) == {}
    saved = [{"adapter": "sphere_single_rigid_body_v1"}] * 14
    assert selected_adapters(task, saved) == selected_adapters(task)
    with pytest.raises(ValueError, match="Unsupported recorded"):
        selected_adapters(task, [{"adapter": "sphere_single_rigid_body_v999"}])
    with pytest.raises(ValueError, match="Unsupported recorded"):
        selected_adapters("stack_blocks", saved)
