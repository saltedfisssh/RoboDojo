"""Motion primitives for the dual X5 on top of RoboDojo's cuRobo v2 planners.

Everything here works in env-local world coordinates (the frame returned by
``LayoutManager.get_instance_pose`` and ``RobotManager.get_real_endpose``).
The executor turns joint paths into 25 Hz 14-D absolute joint actions and runs
them through ``ExpertEnv.take_joint_action``, i.e. the eval control path.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from transforms3d.quaternions import mat2quat, quat2mat

LEFT, RIGHT = 0, 1
ARM_NAMES = ("left", "right")
TABLE_TOP = 0.765
PLAN_TABLE_DROP = 0.009
# Largest per-control-step joint change we command (rad). 25 Hz -> 1.5 rad/s.
MAX_STEP_DELTA = 0.06
FREE_VMAX = 0.10  # rad per control step
FREE_AMAX = 0.02  # rad per control step^2
GRIPPER_FRAMES = 12


class PlanningError(RuntimeError):
    """The expert could not find a feasible motion; the episode is discarded."""


class EpisodeEnd(Exception):
    """The official episode-end logic fired (success or step limit)."""


def top_down_rotation(yaw):
    """link6 frame for a vertical grasp: +X (fingers) down, +Y (jaws) at ``yaw``."""
    x = np.array([0.0, 0.0, -1.0])
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.stack([x, y, np.cross(x, y)], axis=1)


def tilted_rotation(approach, jaw):
    """link6 frame with fingers along ``approach`` and jaws along ``jaw``."""
    x = np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(jaw, float)
    y = y - x * np.dot(x, y)
    y /= np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def yaw_of(quat):
    m = quat2mat(quat)
    return float(np.arctan2(m[1, 0], m[0, 0]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Segment:
    """One annotated stretch of the trajectory (planner primitive)."""

    primitive: str
    text: str
    arm: int | None = None
    obj: str | None = None
    subtask: int = -1
    goal: dict = field(default_factory=dict)
    start_step: int = 0
    end_step: int | None = None
    background: bool = False


class Planner:
    def __init__(self, env, recorder=None):
        self.env = env
        self.rm = env.robot_manager
        self.lm = env.scene_manager.layout_manager
        self.robots = env.arms
        self.solvers = [self.rm.planner[r.robot_name] for r in self.robots]
        self.recorder = recorder
        self.bias = float(self.robots[0].gripper_bias)
        state = env.observe(images=False)["state"]
        self.command = state.copy()
        self.command[[6, 13]] = 1.0
        self.home = self.command.copy()
        self.subtasks = []
        self.segment = None
        self.segments = []
        self.held = [None, None]
        self.queue = {0: None, 1: None}
        self.ignore = set()
        if recorder is not None:
            recorder.bind(self)

    # ------------------------------------------------------------ scene state
    def joints(self, arm):
        return np.asarray(self.rm.get_joint(self.robots[arm])[0], dtype=float)

    def ee(self, arm):
        """link6 pose (7,) in env frame."""
        return np.asarray(self.rm.get_real_endpose(self.robots[arm])[0], dtype=float)

    def tcp(self, arm):
        pose = self.ee(arm)
        return pose[:3] + quat2mat(pose[3:])[:, 0] * self.bias

    def gripper_opening(self, arm):
        r = self.robots[arm]
        q = float(self.rm.get_end_effector_real_val(r)[0][0])
        lo, hi = r.gripper_scale
        return (q - lo) / (hi - lo)

    def labels(self, kind="Rigid"):
        return [rec["label"] for rec in self.lm.get_layout_records(0, kind)]

    def pose(self, label):
        pos, quat = self.lm.get_instance_pose(0, label=label)
        return np.asarray(pos, float).reshape(3).copy(), np.asarray(quat, float).reshape(4).copy()

    def metadata(self, label):
        return self.lm.get_instance_metadata(0, label=label)

    def local_box(self, label):
        v = np.asarray(self.metadata(label)["geometry"]["aligned_bbox"]["vertices"], float)
        return v.min(0), v.max(0)

    def fk_tcp_z(self, arm, q):
        pose = self.fk(arm, q)
        return float(pose[2] + quat2mat(pose[3:])[2, 0] * self.bias)

    def mesh_local(self, label, samples=4000):
        """Surface samples of the object's live USD mesh, in the object frame (cached)."""
        cache = self.__dict__.setdefault("_mesh_cache", {})
        if label in cache:
            return cache[label]
        from isaacsim.core.utils.stage import get_current_stage
        from pxr import Usd, UsdGeom

        obj = self.lm.get_scene_object(0, self.lm.get_instance_name(0, label))
        root = get_current_stage().GetPrimAtPath(obj._prim_path)
        origin = self.env.scene_manager.env_origins[0]
        origin = np.asarray(origin.cpu() if hasattr(origin, "cpu") else origin, float)
        tris = []
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), float)
            if not len(pts):
                continue
            M = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            pts = pts @ M[:3, :3] + M[3, :3] - origin
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            o = 0
            for c in counts:
                face = pts[idx[o : o + c]]
                o += c
                for k in range(1, c - 1):  # fan triangulation
                    tris.append((face[0], face[k], face[k + 1]))
        if not tris:
            cache[label] = None
            return None
        tris = np.asarray(tris)
        area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
        rng = np.random.default_rng(0)
        pick = rng.choice(len(tris), samples, p=area / area.sum())
        w = rng.dirichlet([1, 1, 1], samples)
        world = np.einsum("nk,nkd->nd", w, tris[pick])
        pos, quat = self.pose(label)
        local = (world - pos) @ quat2mat(quat)
        cache[label] = local
        return local

    def mesh_world(self, label):
        local = self.mesh_local(label)
        if local is None:
            return None
        pos, quat = self.pose(label)
        return local @ quat2mat(quat).T + pos

    def world_vertices(self, label):
        lo, hi = self.local_box(label)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        pos, quat = self.pose(label)
        return corners @ quat2mat(quat).T + pos

    def top_z(self, label):
        return float(self.world_vertices(label)[:, 2].max())

    def bottom_z(self, label):
        return float(self.world_vertices(label)[:, 2].min())

    def arm_for(self, xy):
        return LEFT if xy[0] < 0 else RIGHT

    # --------------------------------------------------------- collision world
    def sync_world(self, arm, exclude=(), extra=()):
        """Mirror the live scene into this arm's cuRobo world as cuboids.

        The table uses its true top surface; every rigid/geometry object that is
        not excluded (grasp target, held object) becomes an oriented box.
        """
        from curobo.scene import Scene

        solver = self.solvers[arm]
        base = np.asarray(self.robots[arm].entity_origin_pose, float)
        inv = quat2mat(base[3:]).T
        bias = np.asarray(solver.frame_bias, float)

        def to_base(center, rot, dims):
            c = inv @ (np.asarray(center) - base[:3]) + bias
            return {"dims": [float(max(d, 1e-3)) for d in dims], "pose": [*c.tolist(), *mat2quat(inv @ rot).tolist()]}

        # The X5 base sphere dips 7 mm below the mounting plane (= table top), so
        # the planning table sits PLAN_TABLE_DROP lower to keep the base clear.
        top = TABLE_TOP - PLAN_TABLE_DROP
        cuboids = {"table": to_base([0, -0.05, top - 0.025], np.eye(3), [1.4, 1.1, 0.05])}
        skip = set(exclude) | set(self.ignore) | {h for h in self.held if h}
        for kind in ("Rigid", "Geometry"):
            for label in self.labels(kind):
                if label in skip or label == "camera_stand":
                    continue
                lo, hi = self.local_box(label)
                pos, quat = self.pose(label)
                rot = quat2mat(quat)
                cuboids[label] = to_base(pos + rot @ ((lo + hi) / 2), rot, (hi - lo) + 0.01)
        for i, (center, dims) in enumerate(extra):
            cuboids[f"extra_{i}"] = to_base(center, np.eye(3), dims)
        if len(cuboids) > 32:
            raise PlanningError("Too many obstacles for the preallocated cuRobo cuboid cache")
        model = {"cuboid": cuboids}
        scene = Scene.create(deepcopy(model))
        for consumer in (solver.motion_planner, solver.ik_solver):
            consumer.update_world(scene)

    # ------------------------------------------------------------- kinematics
    def ik(self, arm, ee_pose, near=None, max_delta=None):
        """Collision-checked IK; prefer the solution closest to ``near``."""
        solver = self.solvers[arm]
        r = self.robots[arm]
        near = self.joints(arm) if near is None else np.asarray(near, float)
        from utils.transformer import calculate_target_pose

        target = calculate_target_pose(r.entity_origin_pose, solver.robot_origin_pose, np.asarray(ee_pose, float))
        p, q = solver._trans_from_world_to_base(
            np.asarray(solver.robot_origin_pose, np.float32), np.asarray(target, np.float32)
        )
        goal = solver._build_goal_pose(np.asarray(p) + np.asarray(solver.frame_bias), q)
        ik = solver.ik_solver
        current = solver._build_joint_state(near)
        active = ik.get_active_js(current).position
        n = ik.config.num_seeds
        rng = np.random.default_rng(0)
        seeds = np.repeat(active.detach().cpu().numpy().reshape(1, -1), n, axis=0)
        seeds[1:] += rng.normal(0, 0.25, seeds[1:].shape)
        seeds[n // 2 :] = rng.uniform(-np.pi, np.pi, seeds[n // 2 :].shape)
        lim = ik.kinematics.get_joint_limits().position.detach().cpu().numpy()
        seeds = np.clip(seeds, lim[0], lim[1])
        result = ik.solve_pose(
            goal_tool_poses=goal,
            current_state=current,
            seed_config=active.new_tensor(seeds).unsqueeze(0),
            return_seeds=n,
        )
        if result.success is None or result.js_solution is None:
            return None
        pos = result.js_solution.position.detach().cpu().numpy().reshape(-1, len(result.js_solution.joint_names))
        ok = result.success.detach().cpu().numpy().reshape(-1)
        names = list(result.js_solution.joint_names)
        cand = pos[:, [names.index(j) for j in solver.active_joints_name]]
        dist = np.abs(cand - near)
        # Weight the wrist roll less: it is cheap and symmetric grasps rely on it.
        cost = dist[:, :5].sum(1) + 0.3 * dist[:, 5]
        cost[~ok | ~np.isfinite(cand).all(1)] = np.inf
        if max_delta is not None:
            cost[dist.max(1) > max_delta] = np.inf
        if not np.isfinite(cost).any():
            return None
        return cand[int(np.argmin(cost))].copy()

    def fk(self, arm, q):
        """link6 pose in env frame for joints ``q``."""
        import torch

        solver = self.solvers[arm]
        js = solver._build_joint_state(q)
        kin = solver.ik_solver.compute_kinematics(js)
        pose = kin.tool_poses.to_dict()[solver.ee_link]
        p = pose.position.detach().cpu().numpy().reshape(-1)[:3] - np.asarray(solver.frame_bias)
        qt = pose.quaternion.detach().cpu().numpy().reshape(-1)[:4]
        base = np.asarray(self.robots[arm].entity_origin_pose, float)
        rot = quat2mat(base[3:])
        del torch
        return np.r_[rot @ p + base[:3], mat2quat(rot @ quat2mat(qt))]

    # --------------------------------------------------------------- planning
    def plan_free(self, arm, ee_pose, *, exclude=(), extra=()):
        """Collision-aware cuRobo path to a link6 pose; returns (N,6) at control rate.

        If the full world fails (typically the start pose touches an inflated
        obstacle box next to the gripper), retry without obstacles near the
        gripper, then with the table only. Physical success filtering still
        rejects any episode in which a real collision disturbs the scene.
        """
        near = [
            lbl
            for kind in ("Rigid", "Geometry")
            for lbl in self.labels(kind)
            if np.linalg.norm(self.pose(lbl)[0][:2] - self.tcp(arm)[:2]) < 0.15
        ]
        everything = [lbl for kind in ("Rigid", "Geometry") for lbl in self.labels(kind)]
        for skip in (tuple(exclude), tuple(exclude) + tuple(near), tuple(everything)):
            path = self._plan_free_once(arm, ee_pose, skip, extra)
            if path is not None:
                return path
        return None

    def _plan_free_once(self, arm, ee_pose, exclude, extra):
        self.sync_world(arm, exclude=exclude, extra=extra)
        solver = self.solvers[arm]
        start = self.planned_joints(arm)
        result = solver.plan_path(start, np.asarray(ee_pose, float), self.robots[arm].entity_origin_pose)
        if result["status"] != "Success":
            goal = self.ik(arm, ee_pose, near=start)
            if goal is None:
                return None
            result = solver.plan_joint(start, goal)
            if result["status"] != "Success":
                return None
        return self._resample(np.asarray(result["position"], float), start)

    def plan_joint_goal(self, arm, q_goal, *, exclude=()):
        self.sync_world(arm, exclude=exclude)
        start = self.planned_joints(arm)
        result = self.solvers[arm].plan_joint(start, np.asarray(q_goal, float))
        if result["status"] != "Success":
            return self._resample(np.asarray(self._interp(start, q_goal)), start)
        return self._resample(np.asarray(result["position"], float), start)

    def plan_linear(self, arm, ee_pose, *, step=0.01, start=None):
        """Straight Cartesian move via chained nearby IK (no branch jumps)."""
        start_q = self.planned_joints(arm) if start is None else np.asarray(start, float)
        p0 = self.fk(arm, start_q)
        target = np.asarray(ee_pose, float)
        dist = np.linalg.norm(target[:3] - p0[:3])
        ang = 2 * np.arccos(np.clip(abs(np.dot(p0[3:], target[3:])), 0, 1))
        n = max(1, int(np.ceil(max(dist / step, ang / np.deg2rad(5)))))
        q0, q1 = p0[3:], target[3:] * np.sign(np.dot(p0[3:], target[3:]) or 1)
        path, q = [], start_q
        for a in np.linspace(0, 1, n + 1)[1:]:
            quat = (1 - a) * q0 + a * q1
            pose = np.r_[(1 - a) * p0[:3] + a * target[:3], quat / np.linalg.norm(quat)]
            nxt = self.ik(arm, pose, near=q, max_delta=0.5)
            if nxt is None:
                return None
            path.extend(self._interp(q, nxt))
            q = nxt
        return np.asarray(path)

    def _interp(self, q0, q1):
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        n = max(1, int(np.ceil(np.abs(q1 - q0).max() / MAX_STEP_DELTA)))
        return [q0 + (q1 - q0) * a for a in np.linspace(0, 1, n + 1)[1:]]

    def _resample(self, path, start, vmax=None, amax=None):
        """Retime a dense joint path with a trapezoidal profile at the control rate.

        Speed is measured as the largest joint change per control step; the
        limits sit inside the official teleop distribution (p90 0.12 rad/step).
        """
        vmax = FREE_VMAX if vmax is None else vmax
        amax = FREE_AMAX if amax is None else amax
        pts = np.vstack([np.asarray(start, float)[None], np.asarray(path, float)])
        seg = np.abs(np.diff(pts, axis=0)).max(1)
        s_cum = np.r_[0.0, np.cumsum(seg)]
        total = s_cum[-1]
        if total < 1e-6:
            return pts[-1:].copy()
        # Trapezoid (or triangle) profile in arc length.
        t_acc = vmax / amax
        d_acc = 0.5 * amax * t_acc**2
        if 2 * d_acc > total:
            t_acc = np.sqrt(total / amax)
            vpk, t_flat = amax * t_acc, 0.0
        else:
            vpk, t_flat = vmax, (total - 2 * d_acc) / vmax
        T = 2 * t_acc + t_flat
        n = max(1, int(np.ceil(T)))
        out = []
        for k in range(1, n + 1):
            t = T * k / n
            if t < t_acc:
                d = 0.5 * amax * t**2
            elif t < t_acc + t_flat:
                d = 0.5 * amax * t_acc**2 + vpk * (t - t_acc)
            else:
                r = T - t
                d = total - 0.5 * amax * r**2
            d = min(max(d, 0.0), total)
            i = min(np.searchsorted(s_cum, d, side="right") - 1, len(seg) - 1)
            a = 0.0 if seg[i] < 1e-9 else (d - s_cum[i]) / seg[i]
            out.append(pts[i] + a * (pts[i + 1] - pts[i]))
        return np.asarray(out)

    # Pending plans are composed before execution so the second arm can be
    # planned from where the first arm's queued motion leaves it.
    def planned_joints(self, arm):
        return self.command[7 * arm : 7 * arm + 6].copy()

    # -------------------------------------------------------------- execution
    def begin_subtask(self, text, **info):
        self.subtasks.append({"id": len(self.subtasks), "text": text, **info})

    @property
    def now(self):
        return int(self.env.take_action_cnt[0])

    def stage(self, primitive, text, arm=None, obj=None, **goal):
        """Open a foreground segment (closes the previous foreground one)."""
        sid = self.subtasks[-1]["id"] if self.subtasks else -1
        if self.segment is not None and self.segment.end_step is None:
            self.segment.end_step = self.now
        self.segment = Segment(primitive, text, arm, obj, sid, goal)
        self.segment.start_step = self.now
        self.segments.append(self.segment)

    def background(self, arm, path, primitive, text):
        """Queue a motion for ``arm`` that runs while the other arm works."""
        if path is None or not len(path):
            return
        sid = self.subtasks[-1]["id"] if self.subtasks else -1
        seg = Segment(primitive, text, arm, None, sid, {})
        seg.start_step = self.now
        seg.background = True
        self.segments.append(seg)
        self.queue[arm] = (list(np.asarray(path, float)), seg)

    def drain(self):
        """Finish all queued background motions."""
        while any(self.queue.values()):
            self.step(self.command.copy())

    def step(self, action, foreground=()):
        action = np.asarray(action, float).copy()
        for arm, item in list(self.queue.items()):
            if not item or arm in foreground:
                continue
            path, seg = item
            action[7 * arm : 7 * arm + 6] = path.pop(0)
            action[7 * arm + 6] = 1.0
            if not path:
                self.queue[arm] = None
                seg.end_step = self.now + 1
        if self.recorder is not None:
            self.recorder.add(self.env, action, self.segment, len(self.segments) - 1)
        self.command = action.copy()
        done = self.env.take_joint_action(action)
        if done:
            for seg in self.segments:
                if seg.end_step is None:
                    seg.end_step = self.now
            raise EpisodeEnd()

    def execute(self, paths, grip=None, hold=0):
        """Run joint paths for one or both arms in lockstep; finished arms hold."""
        paths = {a: np.asarray(p, float) for a, p in paths.items() if p is not None and len(p)}
        for a in paths:
            if self.queue.get(a):
                raise PlanningError(f"{ARM_NAMES[a]} arm still has a queued background motion")
        n = max((len(p) for p in paths.values()), default=0)
        for i in range(n):
            action = self.command.copy()
            for a, p in paths.items():
                action[7 * a : 7 * a + 6] = p[min(i, len(p) - 1)]
            if grip:
                for a, g in grip.items():
                    action[7 * a + 6] = g
            self.step(action, foreground=tuple(paths))
        for _ in range(hold):
            self.step(self.command.copy(), foreground=tuple(paths))

    def set_gripper(self, arm, value, frames=GRIPPER_FRAMES):
        for _ in range(frames):
            action = self.command.copy()
            action[7 * arm + 6] = float(value)
            self.step(action, foreground=(arm,))

    def move(self, arm, ee_pose, *, linear=False, exclude=(), hold=0, **kw):
        path = self.plan_linear(arm, ee_pose, **kw) if linear else self.plan_free(arm, ee_pose, exclude=exclude)
        if path is None:
            raise PlanningError(f"{ARM_NAMES[arm]} arm cannot reach {np.round(ee_pose, 3).tolist()}")
        self.execute({arm: path}, hold=hold)

    def home_path(self, arm):
        return self.plan_joint_goal(arm, self.home[7 * arm : 7 * arm + 6])

    def go_home(self, arms=(LEFT, RIGHT)):
        self.drain()
        paths = {a: self.home_path(a) for a in arms}
        self.execute(paths, grip={a: 1.0 for a in arms})
