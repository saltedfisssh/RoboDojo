import pytest

from src.datagen.render_guard import preserve_attributes_during_render


class Attribute:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value


@pytest.mark.parametrize("fails", [False, True])
def test_camera_update_preserves_live_and_checkpoint_restored_conveyor_speeds(fails):
    velocity = Attribute((-0.1, 0.0, 0.0))
    angular = Attribute((0.0, 0.0, 0.0))

    def render():
        velocity.Set((0.0, 0.0, 0.0))
        if fails:
            raise RuntimeError("camera capture failed")
        return "camera frame"

    guarded = preserve_attributes_during_render(render, [velocity, angular])
    for expected in [(-0.1, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.0, 0.0)]:
        # A later checkpoint can use a different speed or intentionally stop.
        velocity.Set(expected)
        if fails:
            with pytest.raises(RuntimeError, match="camera capture failed"):
                guarded()
        else:
            assert guarded() == "camera frame"
        assert velocity.Get() == expected
        assert angular.Get() == (0.0, 0.0, 0.0)
