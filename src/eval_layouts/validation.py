"""Render initialization by invoking the unmodified benchmark EvalEnv.reset."""

from pathlib import Path
import sys

import numpy as np


class InitializationClient:
    """Only the policy's reset RPC is omitted; no model actions are requested."""

    def __init__(self, **kwargs):
        pass

    def call(self, func_name, **kwargs):
        if func_name != "reset":
            raise RuntimeError(f"Unexpected model action during layout initialization: {func_name}")


def make_env(
    task,
    app,
    layout_root,
    set_seed=1000,
    warmup_spp=None,
    render_profile="official",
    planner_cache=None,
    skip_policy_planners=False,
):
    from src.expert.sim import ROOT, build_config

    sys.path.insert(0, str(ROOT / "XPolicyLab"))
    import curobo.runtime  # noqa: F401

    from src.eval_client import eval_env

    eval_env.WsModelClient = InitializationClient
    cfg = build_config(task)
    cfg.eval_cfg.update(
        seed=set_seed,
        eval_layout_root=str(Path(layout_root).resolve()),
        eval_traj_root=str(Path(layout_root).resolve() / "Traj"),
        policy_name="layout_initialization",
        additional_info="render_validation",
    )
    cfg.deploy_cfg.update(port=0, host="localhost")
    if skip_policy_planners:
        for robot in cfg.robot.robots:
            if robot.get("type", "target") != "support":
                robot["need_planner"] = False
    if render_profile == "fast":
        # Supported Isaac render settings for initialization screenshots. Native
        # physics, head-camera pose and image size are retained.
        from omegaconf import OmegaConf

        for key, value in {
            "profile": "rtx",
            "antialiasing_mode": "TAA",
            "rendering_mode": "performance",
            "samples_per_pixel": 1,
            "enable_reflections": False,
            "enable_global_illumination": False,
            "enable_translucency": True,
            "enable_dl_denoiser": False,
            "enable_dlssg": False,
            "dlss_mode": 0,
            "enable_direct_lighting": True,
            "enable_shadows": True,
            "carb_settings": {"/rtx/rendermode": "RaytracedLighting"},
        }.items():
            OmegaConf.update(cfg, f"sim.render.{key}", value, force_add=True)
    if warmup_spp is not None:
        from omegaconf import OmegaConf

        OmegaConf.update(cfg, "sim.render.pathtracing_spp", int(warmup_spp), force_add=True)
    from env.robot_manager.robot_manager import RobotManager

    original_setup = RobotManager._setup_planner
    reused = []

    def setup(manager, robot):
        key = (robot.robot_name, str(robot.curobo_yml_path), float(0.74 - robot.entity_origin_pose[2]))
        if key in planner_cache:
            manager.planner[robot.robot_name] = planner_cache[key]
            manager.ik_solver[robot.robot_name] = planner_cache[key]
            reused.append(robot.robot_name)
        else:
            original_setup(manager, robot)
            if robot.robot_name in manager.planner:
                planner_cache[key] = manager.planner[robot.robot_name]

    if planner_cache is not None:
        RobotManager._setup_planner = setup
    try:
        env = eval_env.create_eval_env(cfg, app)
    finally:
        RobotManager._setup_planner = original_setup
    if render_profile == "fast":
        # RobotManager adds wrist views to the shared config in its constructor.
        # Initialization screenshots only need the calibrated native head view.
        for name in ("cam_left_wrist", "cam_right_wrist"):
            cfg.camera.pop(name, None)
    env.layout_warmup_spp = warmup_spp
    env.layout_render_profile = render_profile
    env.layout_reused_planners = reused
    env.layout_skip_policy_planners = skip_policy_planners
    return env


def initialize_and_render(env, index):
    profile = getattr(env, "render_profile", None)
    if profile is not None and profile.name == "pathtracing" and env.layout_warmup_spp is not None:
        profile.spp = int(env.layout_warmup_spp)
    env.reset(seed=[index])
    profile = env.render_profile
    lm = env.scene_manager.layout_manager
    if not lm.layout_valid[0] or not env.success[0]:
        raise ValueError("Official initialization rejected the layout")
    names = env.camera_manager.camera_names[0]
    camera = next(i for i, name in enumerate(names) if name in ("cam_head", "cam_high"))
    if profile.name == "pathtracing" and profile.spp != 16:
        profile.spp = 16
        env.apply_render_profile(cameras_ready=env.capture_manager.is_ready)
    env.obs_manager.render_for_capture()
    frame = env.capture_manager.step(env_ids=[0], cam_ids=[camera])[0]["rgb"][0]["data"]
    frame = np.asarray(frame)[..., :3].copy()
    if (
        frame.ndim != 3
        or frame.shape[2] != 3
        or min(frame.shape[:2]) < 32
        or not np.isfinite(frame).all()
        or frame.std() < 1
    ):
        raise ValueError(f"Invalid initialization render: shape={frame.shape}")
    counts = {
        kind: len(lm.get_layout_records(0, kind))
        for kind in ("Rigid", "Geometry", "Articulation", "Dynamic", "Garment", "Fluid")
    }
    scene = env.scene_manager
    actual = {
        "Rigid+Dynamic": len(scene._rigid_and_dynamic_objects[0]),
        **{
            kind: len(getattr(scene, f"_{kind.lower()}_objects")[0])
            for kind in ("Articulation", "Geometry", "Garment", "Fluid")
        },
    }
    expected = {
        "Rigid+Dynamic": counts["Rigid"] + counts["Dynamic"],
        **{kind: counts[kind] for kind in ("Articulation", "Geometry", "Garment", "Fluid")},
    }
    if actual != expected:
        raise ValueError(f"Instantiated objects differ from JSON: expected={expected}, actual={actual}")
    return frame, {
        "status": "passed",
        "entrypoint": "src.eval_client.eval_env.EvalEnv.reset",
        "official_stability_checks": 1,
        "physics_gpu_override": 1,
        "camera": names[camera],
        "image_shape": list(frame.shape),
        "initialized_cameras": list(names),
        "image_std": float(frame.std()),
        "counts": counts,
        "instantiated_counts": actual,
        "render": getattr(env, "render_metadata", None),
        "warmup_pathtracing_spp": env.layout_warmup_spp,
        "validation_render_profile": env.layout_render_profile,
        "requested_gpu": getattr(env, "layout_device_id", None),
        "planner_devices": {
            name: str(planner.device_cfg.device) for name, planner in env.robot_manager.planner.items()
        },
        "reused_planners": env.layout_reused_planners,
        "policy_planners_warmed": not env.layout_skip_policy_planners,
        "policy_rpc": "reset stub; no policy inference",
    }
