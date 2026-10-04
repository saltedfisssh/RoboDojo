"""Reusable manipulation skills (pick / place / handover-free transfer).

Grasps are proposed from the live object pose and its bounding box, filtered
by collision-checked IK, and ranked by joint-space cost. Placement compensates
the measured object-in-hand offset, so small grasp slips do not accumulate.
"""

import os

import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import mat2quat, quat2mat

from src.expert.planner import ARM_NAMES, TABLE_TOP, PlanningError, wrap, yaw_of

TILTS = tuple(np.deg2rad([0, 15, 30, 45]))
# Fingertips reach ~8 mm below the nominal TCP; pressing them into the table
# pins the jaws by friction (observed on 15 mm T blocks), so keep this clearance.
MIN_TCP_ABOVE_TABLE = 0.012
PRESHAPE = 0.6  # jaw opening fraction: inner gap 2 * (-0.01 + 0.054 * 0.6) = 4.5 cm
DEBUG = os.environ.get("EXPERT_DEBUG") == "1"


class Grasp:
    def __init__(self, arm, rotation, tcp, q_grasp, q_pre, approach):
        self.arm, self.rotation, self.tcp = arm, rotation, np.asarray(tcp, float)
        self.q_grasp, self.q_pre, self.approach = q_grasp, q_pre, approach

    @property
    def quat(self):
        return mat2quat(self.rotation)


class Skills:
    def __init__(self, planner):
        self.p = planner

    # -------------------------------------------------------------- helpers
    def finger_points(self, ee_pose, opening=1.0):
        """Sample points on both fingers (link6 frame -> world) at a jaw ``opening``.

        cuRobo's X5 model has no finger spheres (link7/8), so grasp poses are
        additionally checked against neighbouring objects with these points.
        """
        # Inner pad face sits at +-q along the jaw axis (gripper reads 0.36 on a
        # 2 cm stem); the finger body is ~12 mm thick beyond it.
        q = -0.01 + 0.054 * opening
        half = q + 0.006
        xs = np.linspace(0.10, 0.155, 5)
        ys = np.linspace(-0.006, 0.006, 2)
        zs = np.linspace(-0.011, 0.011, 3)
        local = np.array([[x, sgn * (half + y), z] for sgn in (-1, 1) for x in xs for y in ys for z in zs])
        rot = quat2mat(ee_pose[3:])
        return local @ rot.T + ee_pose[:3]

    def fingers_clear(self, ee_pose, exclude, opening=1.0, margin=0.003):
        p = self.p
        pts = self.finger_points(ee_pose, opening)
        for kind in ("Rigid", "Geometry"):
            for label in p.labels(kind):
                if label in exclude or label == "camera_stand" or label in p.held:
                    continue
                lo, hi = p.local_box(label)
                pos, quat = p.pose(label)
                local = (pts - pos) @ quat2mat(quat)
                if np.any(np.all((local > lo - margin) & (local < hi + margin), axis=1)):
                    return False
        return True

    def corridor_top(self, label, a_xy, b_xy, pad=0.03):
        """Highest top among other objects near the straight carry line a -> b."""
        p = self.p
        a, b = np.asarray(a_xy, float)[:2], np.asarray(b_xy, float)[:2]
        r_self = float(np.max(np.ptp(p.world_vertices(label)[:, :2], axis=0))) / 2
        top = None
        for kind in ("Rigid", "Geometry"):
            for other in p.labels(kind):
                if other in (label, "camera_stand") or other in p.held:
                    continue
                c = p.pose(other)[0][:2]
                ab = b - a
                t = 0.0 if not ab.any() else float(np.clip(np.dot(c - a, ab) / np.dot(ab, ab), 0, 1))
                r = float(np.max(np.ptp(p.world_vertices(other)[:, :2], axis=0))) / 2
                if np.linalg.norm(c - (a + t * ab)) < r + r_self + pad:
                    z = p.top_z(other)
                    top = z if top is None else max(top, z)
        return top

    def clear_lift(self, label, minimum, goal_xy=None):
        """Lift so the held object's bottom clears objects on the way by 3 cm (capped)."""
        p = self.p
        here = p.pose(label)[0][:2]
        top = self.corridor_top(label, here, here if goal_xy is None else goal_xy)
        need = (top + 0.03 - p.bottom_z(label)) if top is not None else 0.0
        return float(np.clip(need, minimum, 0.15))

    def mesh_grasps(self, label, *, depth=0.009, max_width=0.065, min_width=0.008, band=0.007, k=6):
        """Top-down grasps from the live mesh: (tcp, [yaw, yaw+pi]) best first.

        For each jaw direction and nearby centre, the points within the finger
        band at grasp height must lie between the jaws (width in
        [min_width, max_width]) on both sides of the centre. Preference: close to
        the horizontal centroid, then narrower closings.
        """
        p = self.p
        pts = p.mesh_world(label)
        if pts is None:
            return []
        top = float(pts[:, 2].max())
        h = top - float(pts[:, 2].min())
        z = top - min(depth, 0.5 * h)
        slab = pts[pts[:, 2] > z - 0.006]
        centroid = pts[:, :2].mean(0)
        found = []
        offsets = [np.array([dx, dy]) for dx in np.arange(-0.03, 0.031, 0.005) for dy in np.arange(-0.03, 0.031, 0.005)]
        offsets.sort(key=np.linalg.norm)
        for yaw in np.deg2rad(np.arange(0, 180, 15)):
            u = np.array([np.cos(yaw), np.sin(yaw)])
            v = np.array([-u[1], u[0]])
            for off in offsets:
                c = centroid + off
                rel = slab[:, :2] - c
                inband = slab[np.abs(rel @ v) < band]
                if len(inband) < 10:
                    continue
                s_ = (inband[:, :2] - c) @ u
                lo, hi = s_.min(), s_.max()
                if lo > -0.003 or hi < 0.003 or not (min_width <= hi - lo <= max_width):
                    continue
                centre = c + u * (lo + hi) / 2
                found.append((np.linalg.norm(centre - centroid) + 0.3 * (hi - lo), np.r_[centre, z], yaw))
                break
        found.sort(key=lambda f: f[0])
        return [(tcp, [wrap(yaw), wrap(yaw + np.pi)]) for _, tcp, yaw in found[:k]]

    def ee_from_tcp(self, tcp, rotation):
        return np.r_[np.asarray(tcp, float) - rotation[:, 0] * self.p.bias, mat2quat(rotation)]

    def rotations(self, arm, tcp, yaws, tilts=None):
        """Grasp frames: jaws horizontal along each yaw, fingers pitched outward by each tilt.

        Pitching about the jaw axis keeps both pads on vertical faces while
        moving the wrist toward the robot base, which extends reach.
        """
        tilts = TILTS if tilts is None else tilts
        base = np.asarray(self.p.robots[arm].entity_origin_pose[:3], float)
        out = np.asarray(tcp[:2], float) - base[:2]
        out /= np.linalg.norm(out) + 1e-9
        for tilt in tilts:
            for yaw in yaws:
                y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                h = np.cross(y, [0.0, 0.0, 1.0])
                if np.dot(h[:2], out) < 0:
                    h = -h
                if tilt and np.dot(h[:2], out) < 0.3:
                    continue  # pitching sideways barely helps reach
                x = np.array([0.0, 0.0, -1.0]) * np.cos(tilt) + h * np.sin(tilt)
                yield tilt, yaw, np.stack([x, y, np.cross(x, y)], axis=1)

    def grasp_candidates(
        self, arm, tcp, yaws, *, approach=0.10, tilts=None, place=None, place_yaw=None, lift=0.08, clearance=0.08
    ):
        """Feasible grasps at ``tcp``, least tilt first, then joint cost.

        ``place``: optional object-origin target. The grasp frame is then also
        required to reach the implied lower/carry poses at the destination, so a
        pick is only executed when its place is feasible with the same grip.
        """
        p = self.p
        out = []
        start = p.planned_joints(arm)
        obj = rz = None
        if place is not None:
            obj, obj_quat = p.pose(self._label)
            rz = np.eye(3) if place_yaw is None else euler2mat(0, 0, wrap(place_yaw - yaw_of(obj_quat)))
        centre = tcp(None) if callable(tcp) else tcp
        why = {"fingers": 0, "pre": 0, "grasp": 0, "lift": 0, "place_above": 0, "place": 0}
        self.last_rejections = why
        for tilt, yaw, rot in self.rotations(arm, centre, yaws, tilts):
            t = np.asarray(tcp(yaw) if callable(tcp) else tcp, float).copy()
            t[2] = max(t[2], TABLE_TOP + MIN_TCP_ABOVE_TABLE)
            ee = self.ee_from_tcp(t, rot)
            opening = 1.0
            if getattr(self, "_label", None) and not self.fingers_clear(ee, (self._label,)):
                # Pre-shape the jaws narrower (4.4 cm gap) when a neighbour is close.
                opening = PRESHAPE
                if not self.fingers_clear(ee, (self._label,), opening=opening):
                    why["fingers"] += 1
                    continue
            pre = ee.copy()
            pre[:3] -= rot[:, 0] * approach
            q_pre = p.ik(arm, pre, near=start)
            if q_pre is None:
                why["pre"] += 1
                continue
            q_grasp = p.ik(arm, ee, near=q_pre, max_delta=0.6)
            if q_grasp is None:
                why["grasp"] += 1
                continue
            q_lift = p.ik(arm, self.ee_from_tcp(t + [0, 0, lift], rot), near=q_grasp, max_delta=0.6)
            if q_lift is None:
                why["lift"] += 1
                continue
            if obj is not None:
                goal_tcp = np.asarray(place, float) + rz @ (t - obj)
                grot = rz @ rot
                q_above = p.ik(arm, self.ee_from_tcp(goal_tcp + [0, 0, clearance], grot), near=q_lift)
                if q_above is None:
                    why["place_above"] += 1
                    continue
                if p.ik(arm, self.ee_from_tcp(goal_tcp, grot), near=q_above, max_delta=0.6) is None:
                    why["place"] += 1
                    continue
            reach = np.linalg.norm(t[:2] - np.asarray(p.robots[arm].entity_origin_pose[:2]))
            cost = tilt * 4 + np.abs(q_pre - start)[:5].sum() + 0.3 * abs(q_pre[5] - start[5]) + 4.0 * reach
            g = Grasp(arm, rot, t, q_grasp, q_pre, approach)
            g.opening = opening
            out.append((cost + (0.2 if opening < 1 else 0.0), g))
        out.sort(key=lambda c: c[0])
        return [g for _, g in out]

    def pick_place_grasp(self, label, tcp, yaws, arms, place, exclude=(), **kw):
        """First arm (in order) with a grasp that also reaches ``place``."""
        self._label = label
        try:
            for arm in arms:
                self.p.sync_world(arm, exclude=(label, *exclude))
                cands = self.grasp_candidates(arm, tcp, yaws, place=place, **kw)
                if cands:
                    cands[0].goal_xy = np.asarray(place, float)[:2]
                    return cands[0]
                if DEBUG:
                    print(
                        f"[skills] no grasp {label} arm={arm} place={np.round(place, 3)} rejected={self.last_rejections}",  # noqa: E501
                        flush=True,
                    )
        finally:
            self._label = None
        return None

    def reachable(self, arm, tcp, yaws, *, rises=(0.0, 0.05), tilts=None):
        """Some grasp frame reaches ``tcp`` and the listed heights above it."""
        p = self.p
        for _, _, rot in self.rotations(arm, tcp, yaws, tilts):
            q = p.planned_joints(arm)
            for rise in sorted(rises, reverse=True):
                q = p.ik(arm, self.ee_from_tcp(np.asarray(tcp) + [0, 0, rise], rot), near=q)
                if q is None:
                    break
            else:
                return True
        return False

    def best_grasp(self, label, tcp, yaws, arms, **kw):
        for arm in arms:
            self.p.sync_world(arm, exclude=(label,))
            cands = self.grasp_candidates(arm, tcp, yaws, **kw)
            if cands:
                return cands[0]
        return None

    # ----------------------------------------------------------------- pick
    def pick(self, label, grasp, *, lift=0.10, close=0.0, min_lift=0.03, text=None, replan=None, retries=2):
        """Approach, descend, close and lift; verify the object actually rose.

        On a missed grasp, open, back off and retry with ``replan()`` (a fresh
        grasp from the object's current pose); the failed attempt is labelled
        as a wasted step followed by a recovery, as Argus does for teleop.
        """
        p = self.p
        arm = grasp.arm
        name = ARM_NAMES[arm]
        start_z = p.pose(label)[0][2]
        p.stage("approach", f"move the {name} gripper above the {text or label}", arm, label)
        path = p.plan_free(arm, self._pre_pose(grasp), exclude=(label,))
        if path is None:
            raise PlanningError(f"no approach path to {label}")
        p.execute({arm: path}, grip={arm: getattr(grasp, "opening", 1.0)})
        p.stage("descend", f"lower the {name} gripper around the {text or label}", arm, label)
        path = p.plan_linear(arm, self.ee_from_tcp(grasp.tcp, grasp.rotation), step=0.006)
        if path is None:
            raise PlanningError(f"cannot descend onto {label}")
        for q in path:
            action = p.command.copy()
            action[7 * arm : 7 * arm + 6] = q
            p.step(action, foreground=(arm,))
            # Stop if the hand is blocked (fingers on the object or table).
            if p.tcp(arm)[2] - p.fk_tcp_z(arm, q) > 0.006:
                break
        p.stage("grasp", f"close the {name} gripper on the {text or label}", arm, label)
        p.set_gripper(arm, close, frames=10)
        p.held[arm] = label
        lift = self.clear_lift(label, lift, getattr(grasp, "goal_xy", None))
        p.stage("lift", f"lift the {text or label}", arm, label, object_min_dz=min_lift)
        target = self.ee_from_tcp(grasp.tcp + [0, 0, lift], grasp.rotation)
        path = p.plan_linear(arm, target, step=0.01)
        if path is None:
            target = self.ee_from_tcp(grasp.tcp + [0, 0, min(lift, 0.06)], grasp.rotation)
            path = p.plan_linear(arm, target, step=0.01)
        if path is None:
            raise PlanningError(f"cannot lift {label}")
        p.execute({arm: path})
        dz = p.pose(label)[0][2] - start_z
        if DEBUG:
            x, y = grasp.rotation[:, 0], grasp.rotation[:, 1]
            tilt = np.degrees(np.arccos(np.clip(-x[2], -1, 1)))
            jaw = np.degrees(wrap(np.arctan2(y[1], y[0]) - yaw_of(p.pose(label)[1])))
            print(
                f"[skills] grasp {label}: tilt={tilt:.0f} jaw_rel_obj={jaw:.0f} tcp-obj={np.round(grasp.tcp - p.pose(label)[0], 3)}",  # noqa: E501
                flush=True,
            )
            print(
                f"[skills] lifted {label}: dz={dz:.3f} obj={np.round(p.pose(label)[0], 3)} tcp={np.round(p.tcp(arm), 3)} grip={p.gripper_opening(arm):.2f}",  # noqa: E501
                flush=True,
            )
        if dz >= min_lift:
            return grasp
        p.held[arm] = None
        for seg in p.segments[-4:]:
            if seg.arm == arm and seg.obj == label:
                seg.goal["failed_attempt"] = True
        if replan is None or retries <= 0:
            raise PlanningError(f"{label} slipped during lift (dz={dz:.3f})")
        p.stage("regrasp", f"open the {name} gripper and retry grasping the {text or label}", arm, label, recovery=True)
        p.set_gripper(arm, 1.0, frames=8)
        p.drain()
        new = replan()
        if new is None:
            raise PlanningError(f"no new grasp for {label} after a missed grasp")
        return self.pick(
            label, new, lift=lift, close=close, min_lift=min_lift, text=text, replan=replan, retries=retries - 1
        )

    def _pre_pose(self, grasp):
        ee = self.ee_from_tcp(grasp.tcp, grasp.rotation)
        ee[:3] -= grasp.rotation[:, 0] * grasp.approach
        return ee

    # ---------------------------------------------------------------- place
    def wrist_goal(self, label, arm, target_pos, target_yaw=None, lift=0.0):
        """link6 pose that puts the held object's origin at ``target_pos`` (+``lift``)."""
        p = self.p
        obj_pos, obj_quat = p.pose(label)
        ee = p.ee(arm)
        rot, offset = quat2mat(ee[3:]), ee[:3] - obj_pos
        if target_yaw is not None:
            rz = euler2mat(0, 0, wrap(target_yaw - yaw_of(obj_quat)))
            offset, rot = rz @ offset, rz @ rot
        return np.r_[np.asarray(target_pos, float) + offset + [0, 0, lift], mat2quat(rot)]

    def place(
        self,
        label,
        arm,
        target_pos,
        *,
        target_yaw=None,
        clearance=0.08,
        release_above=0.0,
        text=None,
        dest=None,
        retreat=0.08,
        open_to=1.0,
        overshoot=0.0,
        contact_lag=None,
        precise=False,
    ):
        """Carry the held ``label`` so its origin reaches ``target_pos``.

        ``target_yaw`` (object yaw about +Z) rotates the wrist by the required
        delta. ``precise`` re-measures the in-hand pose while hovering just
        above the goal and corrects position/yaw before the final descent.
        """
        p = self.p
        name = ARM_NAMES[arm]
        if DEBUG:
            print(
                f"[skills] place {label}: obj={np.round(p.pose(label)[0], 3)} ee={np.round(p.ee(arm)[:3], 3)} target={np.round(target_pos, 3)}",  # noqa: E501
                flush=True,
            )
        goal = self.wrist_goal(label, arm, target_pos, target_yaw, release_above)
        above = goal.copy()
        # Keep the carried object's bottom above objects along the carry line.
        bottom_at_goal = p.bottom_z(label) + (np.asarray(target_pos)[2] - p.pose(label)[0][2])
        top = self.corridor_top(label, p.pose(label)[0][:2], np.asarray(target_pos)[:2])
        need = (top + 0.03 - bottom_at_goal) if top is not None else 0.0
        above[2] = goal[2] + float(np.clip(need, clearance, 0.15))
        p.stage(
            "transfer",
            f"carry the {text or label} above {dest or 'the target'}",
            arm,
            label,
            object_position=list(map(float, target_pos)),
            **({"object_yaw": float(target_yaw)} if target_yaw is not None else {}),
        )
        path = p.plan_free(arm, above, exclude=(label,))
        if path is None and above[2] > goal[2] + clearance + 1e-6:
            above[2] = goal[2] + clearance  # safe height unreachable: carry lower
            path = p.plan_free(arm, above, exclude=(label,))
        if path is None:
            raise PlanningError(f"no transfer path for {label}")
        p.execute({arm: path})
        p.stage("lower", f"lower the {text or label} onto {dest or 'the target'}", arm, label)
        if precise:
            hover = 0.015
            for _ in range(3):
                try:
                    p.move(
                        arm,
                        self.wrist_goal(label, arm, target_pos, target_yaw, release_above + hover),
                        linear=True,
                        step=0.006,
                    )
                except PlanningError:
                    break  # correction pose unreachable: descend with the current estimate
                pos, quat = p.pose(label)
                err = np.linalg.norm(pos[:2] - np.asarray(target_pos)[:2])
                yerr = 0.0 if target_yaw is None else abs(wrap(yaw_of(quat) - target_yaw))
                if err < 0.003 and yerr < np.deg2rad(1.0):
                    break
            goal = self.wrist_goal(label, arm, target_pos, target_yaw, release_above)
        if contact_lag is None:
            p.move(arm, goal, linear=True, step=0.004 if precise else 0.008)
        else:
            # Lower (past the nominal goal by ``overshoot``) until the object
            # visibly lags the wrist, i.e. it rests on its support.
            deep = goal.copy()
            deep[2] -= overshoot
            path = p.plan_linear(arm, deep, step=0.004)
            if path is None:
                raise PlanningError(f"cannot lower {label}")
            lag0 = p.ee(arm)[2] - p.pose(label)[0][2]
            for q in path:
                action = p.command.copy()
                action[7 * arm : 7 * arm + 6] = q
                p.step(action, foreground=(arm,))
                if (p.ee(arm)[2] - p.pose(label)[0][2]) - lag0 < -contact_lag:
                    break
        p.stage("release", f"open the {name} gripper to release the {text or label}", arm, label)
        p.set_gripper(arm, open_to, frames=8)
        p.held[arm] = None
        p.stage("retreat", f"lift the {name} gripper away", arm, label)
        ee = p.ee(arm)
        to_base = np.r_[np.asarray(p.robots[arm].entity_origin_pose[:2]) - ee[:2], 0.0]
        to_base /= np.linalg.norm(to_base) + 1e-9
        options = [
            (np.array([0.0, 0.0, 1.0]), retreat),
            (-quat2mat(ee[3:])[:, 0], retreat),
            (np.array([0.0, 0.0, 1.0]), retreat / 2),
            ((to_base + [0, 0, 1.0]) / np.sqrt(2), retreat),
        ]
        for direction, dist in options:
            back = ee.copy()
            back[:3] += direction * dist
            path = p.plan_linear(arm, back, step=0.012)
            if path is not None:
                p.execute({arm: path})
                break
        else:
            back = ee.copy()
            back[2] += retreat
            path = p.plan_free(arm, back, exclude=(label,))
            if path is None:
                raise PlanningError(f"{name} arm cannot retreat from {label}")
            p.execute({arm: path})

    def hold_aside(self, label, arm, xy, height, text=None):
        """Keep the held object lifted at a parking spot clear of the work area."""
        p = self.p
        pos = p.pose(label)[0]
        target = np.r_[xy, max(pos[2], TABLE_TOP + height)]
        goal = self.wrist_goal(label, arm, target)
        p.stage("transfer", f"move the {text or label} aside and hold it", arm, label)
        path = p.plan_free(arm, goal, exclude=(label,))
        if path is None:
            raise PlanningError(f"cannot hold {label} aside")
        p.execute({arm: path})

    def home(self, arms, text=None, background=False):
        p = self.p
        arms = tuple(arms)
        who = " and ".join(ARM_NAMES[a] for a in arms)
        text = text or f"return the {who} arm{'s' if len(arms) > 1 else ''} to the initial pose"
        if background:
            for a in arms:
                p.background(a, p.home_path(a), "home", f"return the {ARM_NAMES[a]} arm to the initial pose")
            return
        p.stage("home", text, arms[0] if len(arms) == 1 else None)
        p.go_home(arms)

    # ------------------------------------------------------- simple helpers
    @staticmethod
    def table_z(p, label):
        """Object-origin height when resting on the table."""
        return TABLE_TOP + (p.pose(label)[0][2] - p.bottom_z(label))
