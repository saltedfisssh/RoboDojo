"""Hardware-aware rendering profiles, independent of Isaac imports at module load."""

from dataclasses import dataclass, field
import json
import os

# Keep the RTX quality preset intact. These settings belong only to the
# compatibility profiles, applied again after camera/Replicator initialization.
AA_LIMITED_OPS = "/rtx-transient/post/aa/limitedOps"
LOW_NOISE_SETTINGS = {
    "/rtx/reflections/enabled": False,
    "/rtx/indirectDiffuse/enabled": False,
    "/rtx/translucency/enabled": False,
    "/rtx-transient/dldenoiser/enabled": False,
    "/rtx-transient/dlssg/enabled": False,
    # Replicator's set_render_rtx_realtime(antialiasing="Off") also clears
    # this flag; otherwise Kit clamps aa/op=0 back to DLSS on every update.
    AA_LIMITED_OPS: False,
    "/rtx/post/aa/op": 0,
    # execMode=0 means Performance, not disabled; aa/op=0 disables DLSS/DLAA.
    "/rtx/post/dlss/execMode": 0,
    "/rtx/directLighting/enabled": True,
    "/rtx/shadows/enabled": True,
}
LIVE_KEYS = (
    "/rtx/rendermode",
    "/rtx/minimal/mode",
    *LOW_NOISE_SETTINGS,
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/pathtracing/optixDenoiser/enabled",
    "/rtx/pathtracing/spp",
    "/rtx/pathtracing/totalSpp",
)


def ray_reconstruction_support(name, capability):
    """Conservative architecture check; unknown hardware retains existing settings.

    https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_architecture_support.html
    Ampere compute (8.0) differs from Ampere RTX (8.6). In particular, do not
    interpret newer architectures such as RTX Blackwell (12.0) as unsupported.
    """
    if capability in ((8, 0), (9, 0)) or "GTX" in name.upper():
        return False
    if capability in ((7, 5), (8, 6), (8, 9), (10, 0), (12, 0)):
        return True
    return None


def renderer_device_names():
    """Hydra reports devices actually participating in rendering."""
    try:
        from omni.hydra.engine.stats import get_device_info

        return [device["description"] for device in get_device_info()]
    except (ImportError, RuntimeError, KeyError):
        return []


def detect_render_gpus(settings):
    """Inspect the rendering device, even when the physics device is CPU.

    Match Hydra's device names to CUDA capabilities instead of assuming Vulkan
    and CUDA ordinals agree (CUDA_VISIBLE_DEVICES can reorder the latter).
    Before Hydra is available, fall back to AppLauncher's selected ordinal.
    """
    import torch

    count = torch.cuda.device_count()
    active = settings.get("/renderer/activeGpu")
    active = 0 if active is None or active < 0 else int(active)
    names = renderer_device_names()
    if names:
        properties = [torch.cuda.get_device_properties(index) for index in range(count)]
        indices = []
        for name in names:
            matches = [index for index, props in enumerate(properties) if props.name == name]
            if not matches:
                raise RuntimeError(f"Rendering GPU {name!r} cannot be matched to a visible CUDA device")
            indices.append(matches[0])
    else:
        indices = list(range(count)) if settings.get("/renderer/multiGpu/enabled") else [active]
    if not count or any(index >= count for index in indices):
        raise RuntimeError(f"Renderer GPU {active} is not visible to CUDA (device count={count})")
    devices = []
    for index in indices:
        props = torch.cuda.get_device_properties(index)
        capability = (props.major, props.minor)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "compute_capability": list(capability),
                "ray_reconstruction": ray_reconstruction_support(props.name, capability),
            }
        )
    return devices


@dataclass
class RenderProfile:
    requested: str
    name: str
    spp: int = 16
    devices: list = field(default_factory=list)
    reason: str = "explicit selection"
    live: dict = field(default_factory=dict)

    @classmethod
    def resolve(cls, config, settings):
        requested = os.environ.get("ROBODOJO_RENDER_PROFILE", config.get("profile", "auto"))
        if requested not in ("auto", "rtx", "low_noise", "minimal", "pathtracing"):
            raise ValueError(f"Unknown render profile {requested!r}; use auto, rtx, low_noise, minimal or pathtracing")
        spp = config.get("pathtracing_spp", 16)
        if isinstance(spp, bool) or not isinstance(spp, int) or spp < 1:
            raise ValueError("render.pathtracing_spp must be a positive integer")
        name, devices, reason = requested, [], "explicit selection"
        if requested == "auto":
            try:
                devices = detect_render_gpus(settings)
                unsupported = any(device["ray_reconstruction"] is False for device in devices)
                name = "low_noise" if unsupported else "rtx"
                if unsupported:
                    reason = "GPU lacks DLSS Ray Reconstruction"
                elif any(device["ray_reconstruction"] is None for device in devices):
                    reason = "unknown GPU architecture; retain RTX configuration"
                else:
                    reason = "retain RTX configuration"
            except Exception as exc:
                name, reason = "rtx", f"GPU detection unavailable; retain RTX configuration: {exc}"
        if name in ("low_noise", "minimal"):
            # Carb accepts arbitrary keys. Probe BEFORE writing, otherwise an
            # unsupported Minimal mode could appear to be successfully enabled.
            if settings.get("/rtx/minimal/mode") is not None:
                name = "minimal"
            else:
                name = "pathtracing"
                reason += "; Minimal unavailable, using PathTracing + OptiX"
        return cls(requested, name, spp, devices, reason)

    def carb_settings(self):
        if self.name == "rtx":
            return {}
        settings = dict(LOW_NOISE_SETTINGS)
        if self.name == "minimal":
            settings.update({"/rtx/rendermode": "MinimalRendering", "/rtx/minimal/mode": 2})
        else:
            settings.update(
                {
                    "/rtx/rendermode": "PathTracing",
                    "/rtx/pathtracing/optixDenoiser/enabled": True,
                    "/rtx/pathtracing/spp": self.spp,
                    "/rtx/pathtracing/totalSpp": self.spp,
                }
            )
        return settings

    def configure(self, config):
        """Run after user overrides; a compatibility profile owns these settings."""
        config = dict(config)
        if self.name == "rtx":
            return config
        config.update(
            {
                "antialiasing_mode": None,
                "enable_reflections": False,
                "enable_global_illumination": False,
                "enable_translucency": False,
                "enable_dl_denoiser": False,
                "enable_dlssg": False,
                "dlss_mode": 0,
                "enable_direct_lighting": True,
                "enable_shadows": True,
            }
        )
        config["carb_settings"] = {**(config.get("carb_settings") or {}), **self.carb_settings()}
        # Initialize render products with the launch renderer first. In Kit
        # 5.1, selecting PathTracing before their first update can leave tiled
        # RGB buffers uninitialized. Switch modes only in apply().
        config["carb_settings"].pop("/rtx/rendermode", None)
        # This transient key may not exist until cameras initialize. IsaacLab
        # rejects unregistered carb_settings, so write it only in apply().
        config["carb_settings"].pop(AA_LIMITED_OPS, None)
        return config

    def apply(self, settings, render):
        """Reapply after render-product creation, then read back after rendering.

        Rendering flushes Kit's asynchronous mode changes without stepping physics.
        RTX profiles only report settings, leaving all existing overrides untouched.
        """
        if self.name != "rtx":
            # A mode switch can reset AA on the next Kit update (5.1 sets it
            # to DLSS). Let that update finish, then restore our settings.
            # Bound retries so an unavailable denoiser never fails silently.
            for _ in range(3):
                for key, value in self.carb_settings().items():
                    if settings.get(key) != value:
                        settings.set(key, value)
                render()
                if self.name == "minimal" and settings.get("/rtx/rendermode") != "MinimalRendering":
                    self.name = "pathtracing"
                    self.reason += "; renderer rejected Minimal, using PathTracing + OptiX"
                elif all(settings.get(key) == value for key, value in self.carb_settings().items()):
                    break
        self.live = {key: settings.get(key) for key in LIVE_KEYS}
        print("[RoboDojo render] " + json.dumps(self.metadata(), sort_keys=True), flush=True)
        mismatches = {
            key: {"expected": value, "actual": settings.get(key)}
            for key, value in self.carb_settings().items()
            if settings.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Render profile {self.name!r} did not take effect: {mismatches}")
        return self.metadata()

    def metadata(self):
        return {
            "requested": self.requested,
            "profile": self.name,
            "reason": self.reason,
            "gpus": self.devices,
            "carb_settings": dict(self.live),
        }
