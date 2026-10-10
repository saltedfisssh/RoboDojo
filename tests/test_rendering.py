"""Rendering policy and initialization regressions without starting Isaac Sim."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from env.environment import rendering


class Settings:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.writes = []

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.writes.append((key, value))
        self.values[key] = value


@pytest.fixture(autouse=True)
def no_profile_override(monkeypatch):
    monkeypatch.delenv("ROBODOJO_RENDER_PROFILE", raising=False)


def gpu(monkeypatch, name, capability):
    monkeypatch.setattr(
        rendering,
        "detect_render_gpus",
        lambda settings: [
            {
                "index": 0,
                "name": name,
                "ray_reconstruction": rendering.ray_reconstruction_support(name, capability),
            }
        ],
    )


@pytest.mark.parametrize(
    "name,capability,profile",
    [
        ("NVIDIA A800-SXM4-80GB", (8, 0), "pathtracing"),
        ("NVIDIA A100", (8, 0), "pathtracing"),
        ("NVIDIA H100", (9, 0), "pathtracing"),
        ("NVIDIA H800", (9, 0), "pathtracing"),
        ("NVIDIA GeForce GTX 1660", (7, 5), "pathtracing"),
        ("NVIDIA GeForce RTX 2080", (7, 5), "rtx"),
        ("NVIDIA GeForce RTX 3090", (8, 6), "rtx"),
        ("NVIDIA GeForce RTX 4090", (8, 9), "rtx"),
        ("NVIDIA GeForce RTX 5090", (12, 0), "rtx"),
        ("NVIDIA B200", (10, 0), "rtx"),
        ("Future GPU", (20, 0), "rtx"),
    ],
)
def test_auto_preserves_rr_hardware_and_selects_compatibility(monkeypatch, name, capability, profile):
    gpu(monkeypatch, name, capability)
    assert rendering.RenderProfile.resolve({}, Settings()).name == profile


def test_5090_preserves_custom_rtx_settings_and_does_not_render(monkeypatch):
    gpu(monkeypatch, "NVIDIA GeForce RTX 5090", (12, 0))
    config = {"antialiasing_mode": "DLAA", "rendering_mode": "quality", "carb_settings": {"/custom": 2}}
    original = deepcopy(config)
    settings = Settings({"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 4})
    profile = rendering.RenderProfile.resolve(config, settings)
    assert profile.configure(config) == original
    profile.apply(settings, lambda: pytest.fail("RTX path must not add render passes"))
    assert settings.writes == []
    assert config == original


def test_selected_render_gpu_is_used_even_with_cpu_physics(monkeypatch):
    import torch

    devices = [SimpleNamespace(name="A800", major=8, minor=0), SimpleNamespace(name="RTX 5090", major=12, minor=0)]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(devices))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: devices[index])
    monkeypatch.setattr(rendering, "renderer_device_names", lambda: [])
    settings = Settings({"/renderer/activeGpu": 1, "/renderer/multiGpu/enabled": False})
    assert rendering.RenderProfile.resolve({}, settings).name == "rtx"
    settings.values["/renderer/activeGpu"] = 0
    assert rendering.RenderProfile.resolve({}, settings).name == "pathtracing"
    settings.values.update({"/renderer/activeGpu": 1, "/renderer/multiGpu/enabled": True})
    assert rendering.RenderProfile.resolve({}, settings).name == "pathtracing"


def test_hydra_device_identity_takes_priority_over_reordered_cuda_indices(monkeypatch):
    import torch

    devices = [SimpleNamespace(name="A800", major=8, minor=0), SimpleNamespace(name="RTX 5090", major=12, minor=0)]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(devices))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: devices[index])
    monkeypatch.setattr(rendering, "renderer_device_names", lambda: ["RTX 5090"])
    settings = Settings({"/renderer/activeGpu": 0, "/renderer/multiGpu/enabled": True})
    profile = rendering.RenderProfile.resolve({}, settings)
    assert profile.name == "rtx"
    assert [device["name"] for device in profile.devices] == ["RTX 5090"]
    monkeypatch.setattr(rendering, "renderer_device_names", lambda: ["A800"])
    assert rendering.RenderProfile.resolve({}, settings).name == "pathtracing"


def test_detection_failure_keeps_existing_configuration_and_is_reported(monkeypatch):
    def fail(settings):
        raise RuntimeError("CUDA unavailable")

    monkeypatch.setattr(rendering, "detect_render_gpus", fail)
    profile = rendering.RenderProfile.resolve({}, Settings())
    assert profile.name == "rtx"
    assert "CUDA unavailable" in profile.reason


def test_environment_override_and_explicit_profiles_skip_detection(monkeypatch):
    monkeypatch.setattr(rendering, "detect_render_gpus", lambda _: pytest.fail("unexpected detection"))
    monkeypatch.setenv("ROBODOJO_RENDER_PROFILE", "rtx")
    assert rendering.RenderProfile.resolve({"profile": "pathtracing"}, Settings()).name == "rtx"
    monkeypatch.delenv("ROBODOJO_RENDER_PROFILE")
    assert rendering.RenderProfile.resolve({"profile": "pathtracing"}, Settings()).name == "pathtracing"


def test_compatibility_overrides_conflicts_without_mutating_user_config():
    config = {
        "profile": "pathtracing",
        "pathtracing_spp": 32,
        "antialiasing_mode": "DLAA",
        "carb_settings": {"/rtx/post/aa/op": 4, "/rtx/rendermode": "RaytracedLighting", "/custom": 7},
    }
    original = deepcopy(config)
    profile = rendering.RenderProfile.resolve(config, Settings())
    resolved = profile.configure(config)
    assert resolved["antialiasing_mode"] is None
    assert resolved["carb_settings"]["/rtx/post/aa/op"] == 0
    assert "/rtx/rendermode" not in resolved["carb_settings"]
    assert resolved["carb_settings"]["/rtx/pathtracing/spp"] == 32
    assert resolved["carb_settings"]["/custom"] == 7
    assert rendering.AA_LIMITED_OPS not in resolved["carb_settings"]
    assert config == original


@pytest.mark.parametrize(
    "minimal_available,rejected,expected",
    [
        (False, False, "PathTracing"),
        (True, False, "MinimalRendering"),
        (True, True, "PathTracing"),
    ],
)
def test_minimal_is_probed_before_writing_and_falls_back_after_live_render(minimal_available, rejected, expected):
    settings = Settings({"/rtx/minimal/mode": 0} if minimal_available else {})
    profile = rendering.RenderProfile.resolve({"profile": "low_noise"}, settings)
    renders = []

    def render():
        renders.append(settings.get("/rtx/rendermode"))
        if rejected and settings.get("/rtx/rendermode") == "MinimalRendering":
            settings.values["/rtx/rendermode"] = "RaytracedLighting"

    metadata = profile.apply(settings, render)
    assert settings.get("/rtx/rendermode") == expected
    assert renders[-1] == expected
    assert metadata["carb_settings"]["/rtx/rendermode"] == expected
    if not minimal_available:
        assert settings.get("/rtx/minimal/mode") is None


def test_camera_reset_overrides_are_reapplied():
    settings = Settings()
    profile = rendering.RenderProfile.resolve({"profile": "pathtracing"}, settings)
    for _ in range(2):
        settings.values.update({"/rtx/rendermode": "RaytracedLighting", "/rtx/post/aa/op": 4})
        metadata = profile.apply(settings, lambda: None)
        assert metadata["carb_settings"]["/rtx/post/aa/op"] == 0
        assert metadata["carb_settings"]["/rtx/pathtracing/optixDenoiser/enabled"] is True


def test_failed_denoiser_is_reported_instead_of_silent_noisy_capture():
    settings = Settings()
    profile = rendering.RenderProfile.resolve({"profile": "pathtracing"}, settings)
    with pytest.raises(RuntimeError, match="optixDenoiser"):
        profile.apply(settings, lambda: settings.values.update({"/rtx/pathtracing/optixDenoiser/enabled": False}))


def test_delayed_mode_switch_cannot_reenable_dlss():
    settings = Settings()
    profile = rendering.RenderProfile.resolve({"profile": "pathtracing"}, settings)
    updates = []

    def render():
        if not updates:
            settings.values["/rtx/post/aa/op"] = 3
        updates.append(True)

    profile.apply(settings, render)
    assert settings.get("/rtx/post/aa/op") == 0
    assert len(updates) == 2
    assert settings.writes.count(("/rtx/rendermode", "PathTracing")) == 1


def test_replicator_aa_restriction_is_removed_before_disabling_aa():
    settings = Settings({rendering.AA_LIMITED_OPS: True})
    profile = rendering.RenderProfile.resolve({"profile": "pathtracing"}, settings)

    def render():
        if settings.get(rendering.AA_LIMITED_OPS):
            settings.values["/rtx/post/aa/op"] = 3

    profile.apply(settings, render)
    assert settings.get("/rtx/post/aa/op") == 0
    assert settings.writes.index((rendering.AA_LIMITED_OPS, False)) < settings.writes.index(("/rtx/post/aa/op", 0))


@pytest.mark.parametrize("config", [{"profile": "typo"}, {"pathtracing_spp": 0}, {"pathtracing_spp": True}])
def test_invalid_configuration_fails_before_simulation(config):
    with pytest.raises(ValueError):
        rendering.RenderProfile.resolve(config, Settings())
