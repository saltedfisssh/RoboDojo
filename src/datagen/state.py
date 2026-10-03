"""Portable physical checkpoints for rigid RoboDojo scenes (no pickle).

InteractiveScene.get_state does NOT contain RoboDojo's externally registered
SingleRigidPrim objects, so they must be captured separately. PhysX solver/contact
warm-start caches are not public state; this is a physical reset, not bitwise replay.
"""

from copy import deepcopy
import random

import numpy as np
import torch

SCHEMA = "robodojo.rigid-state.v1"


def plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Cannot checkpoint {type(value).__name__}; add a task state adapter")


def encode_context(value, parser=None):
    """Preserve tuple-vs-list reward semantics without executable serialization."""
    if callable(value):
        name = getattr(value, "__name__", "")
        if parser is None or name.startswith("_") or getattr(parser, name, None) != value:
            raise TypeError("Only named public methods of this reward parser can be checkpointed")
        return {"__parser_method__": name}
    if isinstance(value, tuple):
        return {"__tuple__": [encode_context(v, parser) for v in value]}
    if isinstance(value, (np.ndarray, torch.Tensor)):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        return {"__ndarray__": array.tolist(), "dtype": str(array.dtype)}
    if isinstance(value, dict):
        return {str(k): encode_context(v, parser) for k, v in value.items()}
    if isinstance(value, list):
        return [encode_context(v, parser) for v in value]
    return plain(value)


def decode_context(value, parser=None):
    if isinstance(value, dict):
        if set(value) == {"__parser_method__"}:
            name = value["__parser_method__"]
            method = getattr(parser, name, None) if parser is not None and not name.startswith("_") else None
            if not callable(method):
                raise ValueError(f"Unknown reward parser method {name!r}")
            return method
        if set(value) == {"__tuple__"}:
            return tuple(decode_context(v, parser) for v in value["__tuple__"])
        if set(value) == {"__ndarray__", "dtype"}:
            return np.asarray(value["__ndarray__"], dtype=value["dtype"])
        return {k: decode_context(v, parser) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_context(v, parser) for v in value]
    return value


def rigid_objects(env):
    lm = env.scene_manager.layout_manager
    objects = {}
    records = lm.get_layout_records(0, "Rigid") + lm.get_layout_records(0, "Articulation")
    for i, rec in enumerate(records):
        identity = rec.get("label") or f"{rec['category']}:{rec['category_idx']}:{i}"
        if identity in objects:
            raise ValueError(f"Duplicate object identity {identity}")
        objects[identity] = (rec, lm.get_scene_object(0, rec["inst_name"]))
    return objects


def terminal_checks(manager):
    """Preserve final physical predicates even after the official state machine latches them."""
    checks = list(manager.check_list[0][-1]) if manager.check_list[0] else []
    checks.extend(manager.final_check_list[0])
    if manager.trigger_check_list[0]:
        condition, payload, _, _ = manager.trigger_check_list[0][-1]
        checks.extend(condition)
        checks.extend(payload)
    parser = manager.func_parser
    return decode_context(encode_context(checks, parser), parser)


class RigidStateAdapter:
    def __init__(self, env, actions):
        self.env, self.actions = env, actions
        if env.num_envs != 1:
            raise ValueError("Checkpoints require one environment")
        from src.datagen.secondary_state import SecondaryState

        self.secondary = SecondaryState(env)
        self.objects = rigid_objects(env)
        self.articulations = {k: obj for k, (_, obj) in self.objects.items() if hasattr(obj, "get_joint_positions")}
        self.visual_state = None
        self.schema = "robodojo.articulated-state.v1" if self.articulations else SCHEMA
        if getattr(env, "interact", False):
            self.schema = "robodojo.interactive-state.v1"
        if self.secondary.enabled:
            self.schema = "robodojo.particle-dynamic-state.v1"
        if self.secondary.cloth_xforms:
            self.schema = "robodojo.cloth-state.v2"

    def capture(self):
        env = self.env
        arrays = {}
        scene = env.sim.scene
        for kind, assets in scene.get_state(is_relative=True).items():
            for name, fields in assets.items():
                for field, value in fields.items():
                    arrays[f"scene/{kind}/{name}/{field}"] = value.detach().cpu().numpy()
        for name, asset in scene.articulations.items():
            for field in ("joint_pos_target", "joint_vel_target", "joint_effort_target"):
                arrays[f"targets/{name}/{field}"] = getattr(asset.data, field).detach().cpu().numpy()
        for identity, (_, obj) in self.objects.items():
            pos, quat = obj.get_world_pose()
            arrays[f"objects/{identity}/pose"] = np.r_[np.asarray(pos) - env.env_origins[0].cpu().numpy(), quat]
            arrays[f"objects/{identity}/velocity"] = np.r_[obj.get_linear_velocity(), obj.get_angular_velocity()]
        articulation_context = {}
        secondary_arrays, secondary_context = self.secondary.capture()
        arrays.update(secondary_arrays)
        if self.articulations:
            from src.datagen.articulations import capture

            for identity, obj in self.articulations.items():
                extra, articulation_context[identity] = capture(obj)
                arrays.update({f"external_articulations/{identity}/{key}": value for key, value in extra.items()})
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise RuntimeError(f"Non-finite simulator state: {name}")
        arrays["command"] = self.actions.command.copy()
        arrays["home"] = self.actions.home.copy()
        arrays["torch_rng"] = torch.get_rng_state().numpy()
        # Only task-independent, JSON-serializable reward state; unsupported custom tasks fail explicitly.
        rm = env.reward_manager
        from src.datagen.asset_adapters import adaptation_signature
        from src.datagen.interaction import capture as capture_interaction

        context = {
            "schema": self.schema,
            "scene_adaptations": adaptation_signature(env),
            "external_articulations": articulation_context,
            "secondary": secondary_context,
            "interaction": encode_context(capture_interaction(env)),
            "objects": {
                k: {"instance": r["inst_name"], "category": r["category"], "index": r["category_idx"]}
                for k, (r, _) in self.objects.items()
            },
            "reward": {
                k: encode_context(v, rm.func_parser)
                for k, v in vars(rm).items()
                if k not in ("env", "func_parser", "num_envs")
            },
            "parser": {
                k: encode_context(getattr(rm.func_parser, k))
                for k in ("pre_state", "robot_origin_endpose", "joint_ratio_transition_state")
            },
            "success": env.success,
            "prev_control": plain(env.robot_manager.control_manager.prev_control),
            "robot_origin_endpose": plain(env.robot_manager.robot_origin_endpose),
            "counters": {k: int(getattr(env.sim, k)) for k in ("_sim_step_counter", "common_step_counter")},
            "episode_length": plain(env.sim.episode_length_buf),
            "python_rng": plain(random.getstate()),
            "numpy_rng": plain(np.random.get_state()),
        }
        return arrays, context

    def restore(self, arrays, context, *, warmup_render=True):
        env, scene = self.env, self.env.sim.scene
        if context["schema"] != self.schema:
            if self.secondary.cloth_xforms and context["schema"] == "robodojo.particle-dynamic-state.v1":
                raise ValueError(
                    "Legacy cloth checkpoint omitted the changing mesh transform; recollect with cloth-state.v2"
                )
            raise ValueError("Unsupported checkpoint schema")
        from src.datagen.asset_adapters import adaptation_signature

        if context.get("scene_adaptations", []) != adaptation_signature(env):
            raise ValueError("Checkpoint scene adaptations differ from the loaded physics scene")
        if set(context["objects"]) != set(self.objects):
            raise ValueError("Checkpoint object set differs from loaded layout")
        renames = {}
        for identity, old in context["objects"].items():
            rec, _ = self.objects[identity]
            if (rec["category"], rec["category_idx"]) != (old["category"], old["index"]):
                raise ValueError(f"Asset mismatch for {identity}")
            renames[old["instance"]] = rec["inst_name"]
        expected, _ = self.capture()
        if set(expected) != set(arrays):
            raise ValueError("Checkpoint entity/target schema differs from scene")
        for name, value in arrays.items():
            if value.shape != expected[name].shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid checkpoint array {name}")
        state = {"articulation": {}, "rigid_object": {}, "deformable_object": {}, "gripper": {}}
        for path, value in arrays.items():
            if path.startswith("scene/"):
                _, kind, name, field = path.split("/")
                state[kind].setdefault(name, {})[field] = torch.as_tensor(value, device=scene.device)
        # Clear actuator history before restoring the saved state and targets.
        scene.reset()
        scene.reset_to(state, is_relative=True)
        for name, asset in scene.articulations.items():
            for field, method in (
                ("joint_pos_target", "set_joint_position_target"),
                ("joint_vel_target", "set_joint_velocity_target"),
                ("joint_effort_target", "set_joint_effort_target"),
            ):
                getattr(asset, method)(torch.as_tensor(arrays[f"targets/{name}/{field}"], device=asset.device))
        for identity, (_, obj) in self.objects.items():
            pose, velocity = arrays[f"objects/{identity}/pose"], arrays[f"objects/{identity}/velocity"]
            obj.set_world_pose(position=pose[:3] + env.env_origins[0].cpu().numpy(), orientation=pose[3:])
            # Isaac Sim 5.1's SingleRigidPrim velocity setters do not convert
            # numpy inputs when SimulationContext uses its torch backend.
            for values, getter, setter in (
                (velocity[:3], obj.get_linear_velocity, obj.set_linear_velocity),
                (velocity[3:], obj.get_angular_velocity, obj.set_angular_velocity),
            ):
                current = getter()
                setter(
                    torch.as_tensor(values, dtype=current.dtype, device=current.device)
                    if isinstance(current, torch.Tensor)
                    else np.asarray(values, dtype=current.dtype)
                )
        if self.articulations:
            from src.datagen.articulations import restore

            for identity, obj in self.articulations.items():
                prefix = f"external_articulations/{identity}/"
                extra = {key.removeprefix(prefix): value for key, value in arrays.items() if key.startswith(prefix)}
                restore(obj, extra, context["external_articulations"][identity])
        if self.secondary.enabled:
            self.secondary.restore(arrays, context["secondary"])
        scene.write_data_to_sim()
        scene.update(0.0)
        self.actions.command = arrays["command"].copy()
        self.actions.home = arrays["home"].copy()

        def remap(value):
            if isinstance(value, dict):
                return {renames.get(k, k): remap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [remap(v) for v in value]
            return value

        for name, value in context["reward"].items():
            setattr(env.reward_manager, name, decode_context(value, env.reward_manager.func_parser))
        for name, value in context["parser"].items():
            setattr(env.reward_manager.func_parser, name, decode_context(remap(value), env.reward_manager.func_parser))
        env.success = list(context["success"])
        env.robot_manager.control_manager.reset()
        env.robot_manager.control_manager.prev_control = deepcopy(context["prev_control"])
        env.robot_manager.robot_origin_endpose = deepcopy(context["robot_origin_endpose"])
        for name, value in context["counters"].items():
            setattr(env.sim, name, value)
        env.sim.episode_length_buf[:] = torch.as_tensor(context["episode_length"], device=scene.device)
        random.setstate((context["python_rng"][0], tuple(context["python_rng"][1]), context["python_rng"][2]))
        rng = context["numpy_rng"]
        np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
        torch.set_rng_state(torch.as_tensor(arrays["torch_rng"], dtype=torch.uint8))
        from src.datagen.interaction import restore as restore_interaction

        restore_interaction(env, decode_context(context.get("interaction")))
        # Lab 2.3.2 forward() only flushes kinematics when Fabric is enabled.
        # For RoboDojo CPU + USD rendering, explicitly publish restored PhysX
        # transforms. Neither API advances simulation time (PhysX 107.3 / Sim 5.1).
        env.sim.sim.physics_sim_view.update_articulations_kinematic()
        env.physics_interface.update_transformations(False, True, False)
        if self.visual_state is None:
            from src.datagen.visual_state import ArticulationVisualState

            self.visual_state = ArticulationVisualState(env, self.articulations)
        self.last_visual_publication = self.visual_state.publish()
        if self.articulations:
            from src.datagen.articulations import restore_sleep

            for identity, obj in self.articulations.items():
                saved = context["external_articulations"][identity]
                if "sleeping" in saved:
                    restore_sleep(obj, saved["sleeping"])
        # A new scene needs material initialization. Rendering successive saved
        # frames in the same scene only needs the normal synchronized capture;
        # repeating the 16-render material warmup for every frame is unnecessary.
        env.datagen_render_warmup = env.datagen_render_warmup or warmup_render

    def errors(self, arrays):
        current, _ = self.capture()
        return {
            name: float(np.max(np.abs(current[name] - value)))
            for name, value in arrays.items()
            if name != "torch_rng" and value.size
        }

    def rollout_errors(self, arrays):
        current, _ = self.capture()
        position, orientation, joint, particles, world_particles = [], [], [], [], []
        for name, value in arrays.items():
            if name.startswith(("objects/", "dynamic/")) and name.endswith("/pose"):
                actual = current[name]
                position.append(float(np.linalg.norm(actual[:3] - value[:3])))
                dot = abs(np.dot(actual[3:], value[3:]) / (np.linalg.norm(actual[3:]) * np.linalg.norm(value[3:])))
                orientation.append(float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1)))))
            elif name.endswith(("/joint_position", "/joint_positions")):
                joint.append(float(np.max(np.abs(current[name] - value))))
            elif name.startswith("particles/") and name.endswith("/world_positions"):
                world_particles.append(float(np.max(np.linalg.norm(current[name] - value, axis=-1))))
            elif name.startswith("particles/") and name.endswith("/local_positions"):
                particles.append(float(np.max(np.linalg.norm(current[name] - value, axis=-1))))
        return {
            "object_position_error_m": max(position, default=0),
            "object_orientation_error_deg": max(orientation, default=0),
            "joint_position_error_rad_or_m": max(joint, default=0),
            "particle_local_position_error": max(particles, default=0),
            "particle_world_position_error_m": max(world_particles, default=0),
        }
