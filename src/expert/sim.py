"""Isaac Sim 5.1 / Isaac Lab 2.3.2 environment for expert data generation.

``launch_app`` must run before anything imports ``isaaclab``/``omni``.
``ExpertEnv`` mirrors ``src/eval_client/eval_env.py`` (reset, layout
stability check, joint-action interpolation, reward/episode-end logic) without
the policy websocket, and reuses one Isaac process for many episodes.
"""

from copy import deepcopy
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAMERA_ALIASES = {"cam_head": "cam_high"}


def launch_app(*, headless=True, cameras=True, device_id=0):
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    if Path("/etc/vulkan/icd.d/nvidia_icd.json").is_file():
        os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args([])
    args.headless = headless
    args.enable_cameras = cameras
    args.device = f"cuda:{device_id}"
    # The container CPU quota is far below os.cpu_count(); Kit sizes its task
    # pools from the latter and the throttled workers stall PhysX stepping.
    threads = int(os.environ.get("EXPERT_CPU_THREADS", "4"))
    physx_threads = int(os.environ.get("EXPERT_PHYSX_THREADS", str(threads)))
    args.kit_args = (
        "--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
        f" --/plugins/carb.tasking.plugin/threadCount={threads}"
        f" --/plugins/omni.tbb.globalcontrol/maxThreadCount={threads}"
        f" --/persistent/physics/numThreads={physx_threads}"
    )
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

    add_zero_delay_kit_args(args)
    return AppLauncher(args).app


def build_config(task):
    """Assemble the env config exactly like ``src/eval_client/main.py``."""
    from omegaconf import OmegaConf

    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "arx_x5.yml"))
    eval_cfg.update(task_name=task, num_envs=1, eval_batch=False, policy_name="expert", seed=0)
    eval_cfg["additional_info"] = ""
    sections = {
        name: load_yaml(os.path.join(ENV_CONFIG_PATH, name, f"{eval_cfg['config'][name]}.yml"))
        for name in ("sim", "scene", "camera", "robot")
    }
    task_cfg = load_yaml(os.path.join(ROOT_DIR, "task", BENCHMARK, "config", f"{task}.yml"))
    cfg = OmegaConf.create({**sections, "task_env": task_cfg, "eval_cfg": eval_cfg, "deploy_cfg": {}})
    OmegaConf.update(cfg, "sim.scene.num_envs", 1, force_add=True)
    cfg = process_randomization(cfg)
    cfg, _ = process_config(cfg, task_name=task)
    OmegaConf.update(cfg, "camera.default_frequency", eval_cfg["observation"]["collect_freq"], force_add=True)
    cfg.sim.seed = [0]
    return cfg


def make_env(task, app, *, cameras=True):
    import importlib

    import curobo.runtime as curobo_runtime  # noqa: F401  (import before planners)
    import torch

    from env.global_configs import BENCHMARK
    from env.observation_manager.obs_manager import ObsManager
    import env.planner_manager.curobo_planner as curobo_adapter

    # One scene, single-pose planning: skip warming the 10-problem batch solver.
    curobo_adapter.BATCH_NUM = 1
    torch.set_num_threads(4)
    for _ in range(16):
        app.update()

    if os.environ.get("EXPERT_PHYSX_GPU", "0") == "0":
        from omni.physx import acquire_physx_interface

        from env.environment.base_env import BaseEnv

        def setup_cpu_physics(self, sim):
            self.physics_interface = acquire_physx_interface()
            self.physics_interface.overwrite_gpu_setting(0)
            self.physics_context = sim.sim.get_physics_context()

        BaseEnv.setup_physics = setup_cpu_physics

    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task, task_class = registry.load_task_class(task)
    cfg = build_config(task)

    class ExpertEnv(task_class):
        def __init__(self, config, app):
            super().__init__(config, app)
            self.task_name = task
            self.eval_cfg = config.eval_cfg
            self.record_images = cameras
            self.obs_manager = ObsManager(
                obs_config=deepcopy(self.eval_cfg["observation"]),
                num_envs=1,
                dt=self.dt,
                task_name=task,
                description_cfg=self.eval_cfg.get("description", {}),
                seeds_per_env=self.env_seed_list,
            )
            self.success = [True]
            self.end_flag = [False]
            self.take_action_cnt = [0]
            self.episode_nums = 1
            self.scene_manager.layout_manager.replay = True

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        # ---------------------------------------------------------------- reset
        def reset_layout(self, seed, layout):
            """Load ``layout`` and run the official settle/stability check.

            Returns False when the official check rejects the layout.
            """
            self.success, self.end_flag, self.take_action_cnt = [True], [False], [0]
            self.episode_nums = 1
            self.scene_manager.layout_manager.set_saved_layout(0, deepcopy(layout))
            super().reset(seed=[int(seed)])
            self.obs_manager.reset()
            self.scene_manager.apply_saved_poses(env_idx_list=[0])
            if not self.scene_manager.layout_manager.layout_valid[0]:
                return False
            stable, _ = self.scene_manager.layout_manager.check_layout_stability(self)
            if not stable or not self.success[0]:
                return False
            if self.record_images:
                from isaacsim.core.utils.stage import is_stage_loading

                # Newly spawned assets stream materials asynchronously; the
                # annotators return empty buffers until loading completes.
                for i in range(200):
                    self.render()
                    if i >= 10 and not is_stage_loading():
                        break
            for idx in range(200):
                self.sim_step(render=False)
                if self.record_images and idx % 5 == 0:
                    self.obs_manager.render_for_capture()
                    self.obs_manager.get_obs()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            self.success, self.end_flag = [True], [False]
            self.run_reward()
            return True

        # ----------------------------------------------------------------- step
        @property
        def arms(self):
            return [r for r in self.robot_manager.robot_list if r.type == "target"]

        def take_joint_action(self, action):
            """Execute one 14-D absolute joint action like EvalEnv.take_action."""
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError(f"Expected finite (14,) joint action, got {action.shape}")
            self.take_action_cnt[0] += 1
            rm = self.robot_manager
            control = {}
            for i, robot in enumerate(self.arms):
                control[rm.process_name(robot.arm_name)] = {"position": action[7 * i : 7 * i + 6].tolist()}
                val = float(np.clip(action[7 * i + 6], 0, 1))
                lo, hi = robot.gripper_scale
                val = val * (hi - lo) + lo if robot.gripper_move["sign"] == 1 else (1 - val) * (hi - lo) + lo
                mimic = robot.gripper_move["mimic"]
                control[rm.process_name(robot.gripper_name)] = {"position": [val, val * mimic[1] + mimic[2]]}
            rm.control_manager.push([0], [self._interpolate(control)])
            while len(rm.control_manager.get_empty(env_idx_list=[0])) == 0:
                task_class.step(self, rm.control_manager.pop([0]))
                self.sim_step(render=False)
            self.reward_manager.step(env_idx_list=[0])
            if getattr(self, "interact", False):
                self.query_support_arm_traj(env_idx=0)
                self.check_support_arm_stable(env_idx=0)
            return self._episode_end()

        def _interpolate(self, control):
            """Same expansion as EvalEnv.process_control_info (80% lerp, 20% hold)."""
            n = int(self.obs_manager.collect_interval)
            seq = [deepcopy(control) for _ in range(n)]
            k = int(np.floor(n * 0.8))
            rm = self.robot_manager
            for robot in rm.robot_list:
                if robot.type != "target":
                    pending = getattr(self, "support_arm_action", None)
                    if pending and pending[0]:
                        for i in range(n):
                            if not pending[0]:
                                break
                            seq[i].update(pending[0].pop(0))
                    continue
                key = rm.process_name(robot.arm_name)
                cur = np.asarray(rm.get_joint(robot, env_idx_list=[0])[0])
                tgt = np.asarray(control[key]["position"])
                for i in range(n):
                    a = (i + 1) / (k + 1) if i < k else 1.0
                    seq[i][key]["position"] = ((1 - a) * cur + a * tgt).tolist()
                key = rm.process_name(robot.gripper_name)
                lo, hi = robot.gripper_scale
                mimic = robot.gripper_move["mimic"]
                cur_g = float(rm.get_end_effector_real_val(robot, env_idx_list=[0])[0][0])
                tgt_g = float(np.clip(control[key]["position"][0], lo, hi))
                for i in range(n):
                    a = (i + 1) / (k + 1) if i < k else 1.0
                    g = float(np.clip((1 - a) * cur_g + a * tgt_g, lo, hi)) if i < k else tgt_g
                    seq[i][key]["position"] = [g, g * mimic[1] + mimic[2]]
            return seq

        def _episode_end(self):
            final = self.take_action_cnt[0] >= self.step_lim or not self.success[0]
            reward = self.reward_manager.get_reward(final_check=final)[0]
            if self.end_flag[0]:
                return True
            if reward > 1 - 1e-3:
                self.end_flag[0], self.success[0] = True, True
            elif final:
                self.end_flag[0], self.success[0] = True, False
            return self.end_flag[0]

        # ---------------------------------------------------------- observation
        def observe(self, images=None):
            images = self.record_images if images is None else images
            om = self.obs_manager
            if images:
                om.render_for_capture()
                obs = om.get_obs([0])[0]
            else:
                capture = om.capture_manager
                om.capture_manager = None
                try:
                    obs = om.get_obs([0])[0]
                finally:
                    om.capture_manager = capture
            state = []
            for robot in self.arms:
                state.extend(np.asarray(obs["state"][f"{robot.arm_name}_joint_state"], dtype=np.float32))
                state.extend(np.asarray(obs["state"][f"{robot.gripper_name}_joint_state"], dtype=np.float32))
            out = {
                "state": np.asarray(state, dtype=np.float32),
                "ee_pose": {
                    k: np.asarray(v, dtype=np.float32) for k, v in obs["state"].items() if k.endswith("ee_pose")
                },
                "instruction": obs["instruction"],
                "images": {},
            }
            for name, cam in obs["vision"].items():
                out["images"][CAMERA_ALIASES.get(name, name)] = np.ascontiguousarray(cam["color"])
            return out

    return ExpertEnv(cfg, app)
