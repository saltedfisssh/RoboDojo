"""Isaac Sim 5.1 / Isaac Lab 2.3.2 adapter. Import AFTER AppLauncher."""

from copy import deepcopy
from time import perf_counter

import numpy as np
from omegaconf import OmegaConf

from env.observation_manager.obs_manager import ObsManager
from src.datagen.layouts import ROOT
from src.datagen.planning_diagnostics import trace_slow_planning as trace_slow_operation
from task.RoboDojo.task_registry import load_task_class
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config


def create_env(app, task, seed, layout, *, cameras=True, env_config=None, planning=True, scene_adaptations=None):
    # Let RTX/OptiX finish first-use initialization before cuRobo captures CUDA graphs.
    # Graph capture can otherwise be invalidated by a concurrent renderer allocation.
    import curobo.runtime as curobo_runtime
    import torch

    import env.planner_manager.curobo_planner as curobo_adapter

    torch.set_num_threads(4)
    # Collection uses one scene and single-pose planning. Avoid allocating and
    # warming the default ten-problem batch solver in each isolated worker.
    curobo_adapter.BATCH_NUM = 1
    if task == "imitate_sorting_sequence":
        # Two full attempts stalled in the first X5 seed-IK graph replay
        # after the Panda demonstration. Use the documented v2 runtime flag
        # before constructing planners, retaining the same collision checks.
        curobo_runtime.cuda_graphs = False

    for _ in range(32):
        app.update()
    torch.cuda.synchronize()
    profile = load_yaml(ROOT / "env_cfg/arx_x5.yml")
    task_path = ROOT / "task/RoboDojo/config" / f"{task}.yml"
    task_cfg = {"Rigid": []} if task == "display_letters" else load_yaml(task_path)
    cfg = OmegaConf.create(
        {
            **{
                name: load_yaml(ROOT / "env_cfg" / name / f"{profile['config'][name]}.yml")
                for name in ("sim", "scene", "robot", "camera")
            },
            "task_env": task_cfg,
            "eval_cfg": profile,
        }
    )
    if env_config is None:
        cfg, _ = process_config(cfg, task)
    else:
        cfg = OmegaConf.create(deepcopy(env_config))
    cfg.sim.scene.num_envs = 1
    cfg.camera.default_frequency = 25
    cfg.sim.seed = [seed]
    # RoboDojo's object wrappers and reward parser use CPU tensors; cuRobo still runs on CUDA.
    cfg.sim.device = "cpu"
    cfg.sim.use_fabric = False
    if task == "display_letters":
        from src.datagen.reconstructed import DisplayLetters

        cls = DisplayLetters
    else:
        _, cls = load_task_class(task)
    if task == "imitate_sorting_sequence":
        from src.datagen.support_demo import training_class

        cls = training_class(cls)
        for robot in cfg.robot.robots:
            if robot.get("type") == "support":
                robot.need_planner = True
    if not planning:
        for robot in cfg.robot.robots:
            robot.need_planner = False
    saved_config = OmegaConf.to_container(cfg, resolve=True)
    from src.datagen.asset_adapters import single_body_assets
    from src.datagen.planning_diagnostics import trace_after, trace_slow_planning

    with single_body_assets(task, scene_adaptations) as adaptations, trace_after(120):
        env = cls(cfg, app)
    try:
        for solver in env.robot_manager.planner.values():
            solver.plan_path = trace_slow_planning(solver.plan_path)
            solver.plan_joint = trace_slow_planning(solver.plan_joint)
            solver.solve_ik_to_joint = trace_slow_planning(solver.solve_ik_to_joint)
        env.datagen_task = task
        if getattr(env, "interact", False):

            def mark_unstable(index):
                env.success[index] = False

            env.mark_env_unstable = mark_unstable
        env.datagen_env_config = saved_config
        env.success = [True]
        env.scene_manager.layout_manager.set_saved_layout(0, deepcopy(layout))
        with single_body_assets(task, scene_adaptations) as reset_adaptations:
            env.reset(seed=[seed])
        env.datagen_scene_adaptations = adaptations + reset_adaptations
        if env.datagen_scene_adaptations:
            print(f"[datagen] Scene adaptations: {env.datagen_scene_adaptations}", flush=True)
        env.scene_manager.apply_saved_poses([0])
        from src.datagen.articulations import initialize_authored_drives

        initialize_authored_drives(env, layout)
        for _ in range(100):
            env.sim_step(render=False)
        env.robot_manager.set_origin_endpose()
        env.robot_manager.set_robot_init_state()
        env.reward_manager.init_state()
        env.run_reward()
        if task == "match_and_pick_from_conveyor":
            from src.datagen.render_guard import guard_constant_conveyor_render

            guard_constant_conveyor_render(env)
        env.obs_manager = ObsManager(profile["observation"], 1, env.dt, task, {"seen": 1}, [seed])
        env.obs_manager.initialize(env)
        if not cameras:
            env.obs_manager.capture_manager = None
        env.datagen_cameras = cameras
        env.datagen_task = task
        env.datagen_task_definition = (
            "reconstructed_from_dataset_instruction" if task == "display_letters" else "official"
        )
        env.datagen_planner_batch_capacity = 1
        env.datagen_curobo_cuda_graphs = bool(curobo_runtime.cuda_graphs)
        env.datagen_render_warmup = True
    except BaseException:
        # A setup error happens before collect can receive the environment.
        # Release its physics resources here so Kit shutdown cannot strand
        # the GPU lease while the parent waits for a long execution timeout.
        env.close()
        raise

    return env


class ActionAdapter:
    """14D absolute joint commands: left six + open fraction, right six + open fraction.

    Match EvalEnv.process_control_info: interpolate for 80% of the 10 physics
    ticks at dt=.004, then hold. Use RoboDojo ControlManager for gripper limiting.
    """

    def __init__(self, env, fps=25):
        self.env = env
        self.rm = env.robot_manager
        self.robots = [r for r in self.rm.robot_list if r.type == "target"]
        if len(self.robots) != 2 or any(len(r.arm_joint_indices) != 6 for r in self.robots):
            raise ValueError("This action schema requires dual X5 with six arm joints per side")
        self.fps = fps
        ticks = 1 / (fps * env.dt)
        if not np.isclose(ticks, round(ticks)) or env.sim.cfg.decimation != 1:
            raise ValueError("FPS must divide the physics rate and decimation must be 1")
        self.ticks = round(ticks)
        self.command = self.measured_state()
        self.command[[6, 13]] = 1
        self.home = self.command.copy()
        self.timings = dict(controls=0.0, commanding=0.0, simulation=0.0, reward=0.0)

    def measured_state(self):
        state = []
        for r in self.robots:
            state.extend(self.rm.get_joint(r)[0])
            val = self.rm.get_end_effector_real_val(r)[0][0]
            opening = (val - r.gripper_scale[0]) / (r.gripper_scale[1] - r.gripper_scale[0])
            state.append(opening if r.gripper_move["sign"] == 1 else 1 - opening)
        return np.asarray(state, dtype=np.float32)

    def observation_state(self):
        # Match the official ObsManager convention: gripper state is previous commanded opening.
        state = self.measured_state()
        for i, r in enumerate(self.robots):
            previous = self.rm.control_manager.prev_control[0][self.rm.process_name(r.gripper_name)]["position"][0]
            opening = (previous - r.gripper_scale[0]) / (r.gripper_scale[1] - r.gripper_scale[0])
            state[i * 7 + 6] = opening if r.gripper_move["sign"] == 1 else 1 - opening
        return state

    @trace_slow_operation
    def apply(self, action):
        stamp = perf_counter()
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected a finite (14,) absolute joint action")
        if np.any(action[[6, 13]] < 0) or np.any(action[[6, 13]] > 1):
            raise ValueError("Gripper open fractions must be in [0,1]")
        controls = [{} for _ in range(self.ticks)]
        interp_count = int(np.floor(self.ticks * 0.8))
        for i, r in enumerate(self.robots):
            current = self.rm.get_joint(r)[0]
            current_gripper = self.rm.get_end_effector_real_val(r)[0][0]
            target = action[i * 7 : i * 7 + 6]
            opening = action[i * 7 + 6]
            if r.gripper_move["sign"] != 1:
                opening = 1 - opening
            target_gripper = r.gripper_scale[0] + opening * (r.gripper_scale[1] - r.gripper_scale[0])
            for step in range(self.ticks):
                alpha = (step + 1) / (interp_count + 1) if step < interp_count else 1.0
                arm = (1 - alpha) * current + alpha * target
                val = float(np.clip((1 - alpha) * current_gripper + alpha * target_gripper, *r.gripper_scale))
                controls[step][self.rm.process_name(r.arm_name)] = {"position": arm, "velocity": np.zeros(6)}
                controls[step][self.rm.process_name(r.gripper_name)] = {
                    "position": [val, val * r.gripper_move["mimic"][1] + r.gripper_move["mimic"][2]],
                    "velocity": np.zeros(2),
                }
        if getattr(self.env, "interact", False):
            pending = self.env.support_arm_action[0]
            for control in controls:
                if pending:
                    control.update(pending.pop(0))
        self.rm.control_manager.push([0], [controls])
        self.timings["controls"] += perf_counter() - stamp
        for _ in range(self.ticks):
            stamp = perf_counter()
            self.env.step(self.rm.control_manager.pop([0]))
            now = perf_counter()
            self.timings["commanding"] += now - stamp
            self.env.sim_step(render=False)
            self.timings["simulation"] += perf_counter() - now
        stamp = perf_counter()
        self.command = action.copy()
        self.env.reward_manager.step(env_idx_list=[0])
        if getattr(self.env, "interact", False):
            self.env.query_support_arm_traj(env_idx=0)
            self.env.check_support_arm_stable(env_idx=0)
        self.timings["reward"] += perf_counter() - stamp

    def images(self):
        if not self.env.datagen_cameras:
            return {}
        if self.env.datagen_render_warmup:
            from isaacsim.core.utils.stage import is_stage_loading

            # First render requests material/shader loads. Subsequent render-only
            # updates complete those requests and populate all three camera buffers.
            for _ in range(16):
                self.env.render()
            for _ in range(120):
                if not is_stage_loading():
                    break
                self.env.render()
            else:
                raise RuntimeError("Scene assets did not finish loading during camera warmup")
            self.env.datagen_render_warmup = False
        self.env.obs_manager.render_for_capture()
        obs = self.env.obs_manager.get_obs([0])[0]
        return {
            ("cam_high" if name == "cam_head" else name): np.asarray(cam["color"]).copy()
            for name, cam in obs["vision"].items()
        }
