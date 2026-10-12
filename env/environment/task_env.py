import builtins
from contextlib import contextmanager
from typing import Any, List

from env.camera_manager.camera_manager import CameraManager
from env.environment.base_env import BaseEnv
from env.robot_manager.control_manager import ControlSeq
from env.robot_manager.robot_manager import RobotManager
from env.scene_manager.scene_manager import SceneManager


class TaskEnv(BaseEnv):
    def __init__(self, config, app, **kwargs):
        self.sim_config = config.sim
        self.scene_config = config.scene
        self.num_envs = self.sim_config.scene.num_envs
        self.env_spacing = self.sim_config.scene.env_spacing
        self.device = self.sim_config.get("device", "cpu")
        self.dt = self.sim_config.dt

        # RobotManager must be created before BaseEnv.__init__ because it
        # provides the InteractiveSceneCfg used to launch the simulation.
        self.robot_config = config.robot
        self.robot_manager = RobotManager(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
            config=self.robot_config,
            dt=self.dt,
            device=self.device,
        )
        self._interactiveSceneCfg = self.robot_manager._get_SceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing
        )
        super().__init__(self.sim_config, app=app, interactiveSceneCfg=self._interactiveSceneCfg, **kwargs)
        self.env_seed_list = self.env_seeds

        self.scene_manager = SceneManager(
            self.num_envs,
            self.device,
            self.sim_config.scene.env_spacing,
            use_fabric=self.use_fabric,
            seeds_per_env=self.env_seed_list,
            scene_config=self.scene_config,
            task_config=config.task_env,
        )

        self.robot_manager._load_camera_cfg(config.camera)
        self.capture_config = config.camera
        self.camera_config = config.camera
        self.camera_manager = CameraManager(
            self.num_envs,
            self.camera_config,
            self.device,
            seeds_per_env=self.env_seed_list,
        )
        from env.camera_manager.capture.tiled_capture_manager import TiledCaptureManager

        self.capture_manager = TiledCaptureManager(
            self.num_envs,
            self.capture_config,
            self.camera_manager,
            self.device,
        )

    def _setup_scene(self, sim):
        self.robot_manager.initialize(sim)
        super()._setup_scene(sim)

    def _post_setup_scene(self, sim):
        self.scene_manager.initialize(sim)
        super()._post_setup_scene(sim)
        self.camera_manager.initialize(sim)
        self.capture_manager.initialize(sim)
        self.camera_manager.post_init()

    def update_seed(self, seed: Any | None = None):
        super().update_seed(seed)
        if hasattr(self, "scene_manager") and self.scene_manager is not None:
            self.scene_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "robot_manager") and self.robot_manager is not None:
            self.robot_manager.update_env_seeds(self.env_seeds)
        if hasattr(self, "camera_manager") and self.camera_manager is not None:
            self.camera_manager.update_env_seeds(self.env_seeds)

    @contextmanager
    def _stopped_physics(self):
        """Keep one native owner while safely entering a stopped USD edit phase."""
        if self.sim is None:
            yield None
            return
        native = self.sim.sim
        stop_guard = native._disable_app_control_on_stop_handle
        native._disable_app_control_on_stop_handle = True
        try:
            native.stop()
            if not native.is_stopped() or native.physics_sim_view is not None:
                raise RuntimeError("Native STOP did not release the shared physics view")
            yield native
        finally:
            native._disable_app_control_on_stop_handle = stop_guard

    def reset(self, seed: Any | None = None, options: dict[str, Any] | None = None):
        self.update_seed(seed)
        if self.sim is None:
            self.launch_sim(self.config)
        self.capture_manager.destroy()
        with self._stopped_physics() as native:
            self.scene_manager.reload_scene()
            native.reset(soft=False)
            callback_error = builtins.ISAACLAB_CALLBACK_EXCEPTION
            if callback_error is not None:
                builtins.ISAACLAB_CALLBACK_EXCEPTION = None
                raise callback_error
            view = native.physics_sim_view
            if view is None or not view.is_valid:
                raise RuntimeError("Native hard reset did not create a valid shared physics view")
            self.sim.reset(options=options)
            self.scene_manager.post_reload_scene()
        self.robot_manager.reset()
        for _ in range(300):
            self.sim_step(render=False)
        self.camera_manager.reset()
        self.capture_manager.initialize(self.sim)
        self.capture_manager.reset()
        self.apply_render_profile(cameras_ready=self.capture_manager.is_ready)

    def step(self, meta_control_list: List[ControlSeq]):
        self.robot_manager.control_robot(meta_control_list=meta_control_list)

    def close(self):
        with self._stopped_physics():
            self.capture_manager.destroy()
            self.camera_manager.destroy()
            self.robot_manager.close()
            self.scene_manager.close()
            super().close()
