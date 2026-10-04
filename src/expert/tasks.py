"""Task programs. Each ``fn(planner, skills)`` drives one episode or raises PlanningError.

Programs read the true scene state, choose an arm assignment whose grasps are
verified (IK at pick, lift, carry and place) before any motion, then execute.
"""

import numpy as np
from transforms3d.quaternions import mat2quat

from src.expert.planner import TABLE_TOP, PlanningError, wrap, yaw_of

TASKS = {}


def task(name):
    def register(fn):
        TASKS[name] = fn
        return fn

    return register


def fail(msg):
    raise PlanningError(msg)


def box_yaws(yaw, step=np.pi / 2):
    """Jaw directions aligned with a box's faces (all wrist alternatives)."""
    return [wrap(yaw + k * step) for k in range(int(round(2 * np.pi / step)))]


def arm_order(p, xy):
    first = p.arm_for(xy)
    return (first, 1 - first)


def centrality(p, label):
    pos = p.pose(label)[0]
    return abs(pos[0]) + 0.5 * abs(pos[1] + 0.15)


# --------------------------------------------------------------------- stacking
class StackSpec:
    """How to grasp a piece and where its origin goes when stacked on a support."""

    place_kw = {}
    lift = 0.08
    text = "object"
    max_carry = 0.45  # m; longer one-arm carries reach the X5 workspace edge

    def __init__(self, p):
        self.p = p
        self.rest = {lbl: p.pose(lbl)[0][2] - p.bottom_z(lbl) for lbl in p.labels("Rigid")}

    def grasp(self, label):  # -> (tcp or tcp(yaw), yaws)
        raise NotImplementedError

    def target(self, label, support_xy, support_top, level):  # -> object-origin target
        raise NotImplementedError

    def height(self, label):
        """Increase of the stack top when ``label`` is added."""
        return float(np.ptp(self.p.world_vertices(label)[:, 2]))


class BlockSpec(StackSpec):
    text = "block"

    def grasp(self, label):
        pos, quat = self.p.pose(label)
        lo, hi = self.p.local_box(label)
        tcp = pos.copy()
        tcp[2] = self.p.top_z(label) - min(0.012, float(hi[2] - lo[2]) / 2)
        return tcp, box_yaws(yaw_of(quat))

    def target(self, label, support_xy, support_top, level):
        return np.r_[support_xy, support_top + self.rest[label] + 0.004]


class BowlSpec(StackSpec):
    text = "bowl"
    lift = 0.10
    max_carry = 0.32
    # Lower into the support bowl until the held bowl stops descending.
    place_kw = {"contact_lag": 0.004, "overshoot": 0.04, "clearance": 0.06}

    def grasp(self, label):
        pos, _ = self.p.pose(label)
        lo, hi = self.p.local_box(label)
        r = float(min(hi[0] - lo[0], hi[1] - lo[1])) / 2 - 0.008
        z = self.p.top_z(label) - 0.015

        def tcp(yaw):
            if yaw is None:
                return np.r_[pos[:2], z]
            return np.r_[pos[:2] + r * np.array([np.cos(yaw), np.sin(yaw)]), z]

        return tcp, [np.deg2rad(a) for a in range(-180, 180, 30)]

    def target(self, label, support_xy, support_top, level):
        # Nominal rim-to-rim seat; contact stopping finds the real one.
        return np.r_[support_xy, support_top - 0.025 + self.rest[label]]

    def height(self, label):
        return 0.012


def plan_stack(p, s, labels, spec):
    """Choose the cheapest feasible base: in place, or first relocated to the centre.

    Every move must have a grasp that reaches its destination with the same
    grip. Cost = total carry distance + 0.25 per extra (relocation) move.
    Returns (base, [(label, arm)], centre_or_None).
    """

    def feasible(base_xy, base_top, movers):
        order, top, cost = [], base_top, 0.0
        for level, label in enumerate(movers, start=1):
            d = float(np.linalg.norm(p.pose(label)[0][:2] - base_xy))
            if d > spec.max_carry:
                return None, None
            tcp, yaws = spec.grasp(label)
            target = spec.target(label, base_xy, top, level)
            g = s.pick_place_grasp(label, tcp, yaws, arm_order(p, p.pose(label)[0]), target, lift=spec.lift)
            if g is None:
                return None, None
            order.append((label, g.arm))
            top += spec.height(label)
            cost += d
        return order, cost

    candidates = []
    for base in labels:
        bpos = p.pose(base)[0]
        movers = sorted((lbl for lbl in labels if lbl != base), key=lambda lbl: abs(p.pose(lbl)[0][0] - bpos[0]))
        candidates.append((sum(np.linalg.norm(p.pose(lbl)[0][:2] - bpos[:2]) for lbl in movers), base, None, movers))
        for cxy in ((0.0, -0.15), (0.0, -0.20), (-0.05, -0.15), (0.05, -0.15)):
            c = np.asarray(cxy)
            dist = np.linalg.norm(bpos[:2] - c) + sum(np.linalg.norm(p.pose(lbl)[0][:2] - c) for lbl in movers) + 0.25
            candidates.append((dist, base, c, movers))
    for _, base, cxy, movers in sorted(candidates, key=lambda c: c[0]):
        bpos = p.pose(base)[0]
        if cxy is None:
            order, _ = feasible(bpos[:2], p.top_z(base), movers)
            if order is not None:
                return base, order, None
            continue
        if np.linalg.norm(bpos[:2] - cxy) > spec.max_carry:
            continue
        centre = np.r_[cxy, bpos[2]]
        tcp, yaws = spec.grasp(base)
        if s.pick_place_grasp(base, tcp, yaws, arm_order(p, bpos), centre, lift=spec.lift) is None:
            continue
        order, _ = feasible(cxy, p.top_z(base) - bpos[2] + centre[2], movers)
        if order is not None:
            return base, order, centre
    return None


def gather(p, s, labels, spec, noun):
    """Bring outlying pieces to free spots nearer the centre, each with its own arm.

    Returns True if anything moved. Used when no direct stacking plan exists.
    """
    moved = False
    r = {lbl: float(np.max(np.ptp(p.world_vertices(lbl)[:, :2], axis=0))) / 2 for lbl in labels}
    for label in sorted(labels, key=lambda lbl: -abs(p.pose(lbl)[0][0])):
        pos = p.pose(label)[0]
        if abs(pos[0]) < 0.22:
            continue
        side = np.sign(pos[0])
        arm = 0 if side < 0 else 1
        for spot in ((0.14, -0.15), (0.14, -0.21), (0.10, -0.10), (0.20, -0.12), (0.08, -0.20)):
            xy = np.array([side * spot[0], spot[1]])
            if any(np.linalg.norm(p.pose(o)[0][:2] - xy) < r[label] + r[o] + 0.02 for o in labels if o != label):
                continue
            target = np.r_[xy, TABLE_TOP + spec.rest[label] + 0.003]
            tcp, yaws = spec.grasp(label)
            g = s.pick_place_grasp(label, tcp, yaws, (arm,), target, lift=spec.lift)
            if g is None:
                continue
            p.begin_subtask(
                f"move the {noun} {label} closer to the centre", object=label, target=[float(v) for v in xy]
            )
            p.drain()
            s.pick(
                label,
                g,
                lift=spec.lift,
                text=noun,
                replan=lambda: s.pick_place_grasp(label, *spec.grasp(label), (arm,), target, lift=spec.lift),
            )
            s.place(label, g.arm, target, text=noun, dest="the middle of the table", **spec.place_kw)
            s.home([g.arm], background=True)
            moved = True
            break
    return moved


def run_stack(p, s, labels, spec, *, noun):
    plan = plan_stack(p, s, labels, spec)
    if plan is None and gather(p, s, labels, spec, noun):
        p.drain()
        plan = plan_stack(p, s, labels, spec)
    if plan is None:
        fail(f"no base/arm assignment reaches every {noun}")
    base, order, centre = plan
    moves = []
    if centre is not None:
        moves.append((base, arm_order(p, p.pose(base)[0])[0], "centre"))
    moves += [(label, arm, None) for label, arm in order]
    top_label = base
    level = 0
    for k, (label, arm, kind) in enumerate(moves):
        if kind == "centre":
            p.begin_subtask(f"move the {noun} {base} to the centre of the table", object=base, target="table centre")
            tcp, yaws = spec.grasp(base)
            target = centre
            dest = "the centre of the table"
        else:
            level += 1
            p.begin_subtask(f"stack {label} on {top_label}", object=label, target=top_label)
            tcp, yaws = spec.grasp(label)
            target = spec.target(label, p.pose(base)[0][:2], p.top_z(top_label), level)
            dest = f"the {top_label}"
        g = s.pick_place_grasp(
            label, tcp, yaws, (arm, 1 - arm) if not p.queue.get(1 - arm) else (arm,), target, lift=spec.lift
        )
        if g is None:
            fail(f"no grasp of {label} reaches {dest}")
        s.pick(
            label,
            g,
            lift=spec.lift,
            text=noun,
            replan=lambda: s.pick_place_grasp(label, *spec.grasp(label), (g.arm,), target, lift=spec.lift),
        )
        s.place(label, g.arm, target, text=noun, dest=dest, **spec.place_kw)
        nxt = moves[k + 1][1] if k + 1 < len(moves) else None
        if nxt is not None and nxt != g.arm:
            # The other arm starts its approach while this one parks.
            s.home([g.arm], background=True)
        if kind is None:
            top_label = label
    p.drain()
    s.home([0, 1])
    return True


@task("stack_blocks")
def stack_blocks(p, s):
    return run_stack(p, s, ["block_0", "block_1", "block_2"], BlockSpec(p), noun="block")


@task("stack_bowls")
def stack_bowls(p, s):
    return run_stack(p, s, ["bowl0", "bowl1", "bowl2"], BowlSpec(p), noun="bowl")


# ---------------------------------------------------------------------- swap_T
@task("swap_T")
def swap_T(p, s):
    poses = {lbl: p.pose(lbl) for lbl in ("t0", "t1")}
    goal = {"t0": poses["t1"], "t1": poses["t0"]}
    # First piece is lifted and parked aside, second goes straight to the first's
    # spot, then the parked one is placed. Try both orders / arm assignments.
    for first, second in (("t0", "t1"), ("t1", "t0")):
        plan = []
        for label in (first, second):
            tgt_pos, tgt_quat = goal[label]
            found = None
            for tcp, yaws in s.mesh_grasps(label):
                g = s.pick_place_grasp(
                    label,
                    tcp,
                    yaws,
                    arm_order(p, poses[label][0]),
                    tgt_pos,
                    place_yaw=yaw_of(tgt_quat),
                    lift=0.06,
                    exclude=("t0", "t1"),
                )
                if g is not None and (not plan or g.arm != plan[0][1].arm):
                    found = (label, g)
                    break
            if found is None:
                break
            plan.append(found)
        if len(plan) == 2:
            break
    else:
        return swap_T_via_buffer(p, s, poses, goal)
    (a_label, ga), (b_label, gb) = plan
    names = {"t0": "first T block", "t1": "second T block"}
    p.begin_subtask(f"pick up the {names[a_label]} and move it aside", object=a_label, target="aside")

    def regrasp(label, arm):
        tgt_pos, tgt_quat = goal[label]
        for tcp, yaws in s.mesh_grasps(label):
            g = s.pick_place_grasp(
                label, tcp, yaws, (arm,), tgt_pos, place_yaw=yaw_of(tgt_quat), lift=0.06, exclude=("t0", "t1")
            )
            if g is not None:
                return g
        return None

    s.pick(a_label, ga, lift=0.06, text="T-shaped block", replan=lambda: regrasp(a_label, ga.arm))
    side = -1 if ga.arm == 0 else 1
    s.hold_aside(a_label, ga.arm, (side * 0.22, -0.12), 0.10, text="T-shaped block")
    p.begin_subtask(
        f"place the {names[b_label]} at the {names[a_label]}'s original pose", object=b_label, target=a_label
    )
    tcp_b = gb.tcp
    gb2 = (
        s.pick_place_grasp(
            b_label,
            tcp_b,
            [yaw_of(mat2quat(gb.rotation))],
            (gb.arm,),
            goal[b_label][0],
            place_yaw=yaw_of(goal[b_label][1]),
            lift=0.06,
        )
        or gb
    )
    s.pick(b_label, gb2, lift=0.06, text="T-shaped block", replan=lambda: regrasp(b_label, gb2.arm))
    s.place(
        b_label,
        gb2.arm,
        goal[b_label][0],
        target_yaw=yaw_of(goal[b_label][1]),
        text="T-shaped block",
        dest=f"the {names[a_label]}'s original spot",
        precise=True,
    )
    s.home([gb2.arm], background=True)
    p.begin_subtask(
        f"place the {names[a_label]} at the {names[b_label]}'s original pose", object=a_label, target=b_label
    )
    s.place(
        a_label,
        ga.arm,
        goal[a_label][0],
        target_yaw=yaw_of(goal[a_label][1]),
        text="T-shaped block",
        dest=f"the {names[b_label]}'s original spot",
        precise=True,
    )
    p.drain()
    s.home([0, 1])
    return True


# ---------------------------------------------------------------- rearrangement
class Rearranger:
    """Move objects to goal positions with free-spot checks, relays and buffers.

    ``spec`` supplies grasps and origin heights (StackSpec-like: grasp(label),
    rest[label]). Footprints are circles of the object's horizontal half-extent.
    """

    def __init__(self, p, s, spec, labels, *, noun, max_moves=12):
        self.p, self.s, self.spec, self.labels, self.noun = p, s, spec, list(labels), noun
        self.max_moves = max_moves
        self.radius = {lbl: float(np.max(np.ptp(p.world_vertices(lbl)[:, :2], axis=0))) / 2 for lbl in labels}
        self.moves = 0

    def xy(self, label):
        return self.p.pose(label)[0][:2]

    def free(self, label, xy, ignore=(), margin=0.012):
        for other in self.p.labels("Rigid"):
            if other in (label, *ignore):
                continue
            r = self.radius.get(other, 0.04)
            if np.linalg.norm(self.xy(other) - xy) < self.radius[label] + r + margin:
                return False
        return True

    def target(self, label, xy):
        return np.r_[xy, TABLE_TOP + self.spec.rest[label] + 0.003]

    def grasp_to(self, label, xy, arms):
        tcp, yaws = self.spec.grasp(label)
        return self.s.pick_place_grasp(label, tcp, yaws, arms, self.target(label, xy), lift=self.spec.lift)

    def move(self, label, xy, grasp, text):
        self.moves += 1
        if self.moves > self.max_moves:
            fail("too many rearrangement moves")
        p, s = self.p, self.s
        p.begin_subtask(text, object=label, target=[float(v) for v in xy])
        if p.queue.get(grasp.arm):
            p.drain()
        s.pick(label, grasp, lift=self.spec.lift, text=self.noun, replan=lambda: self.grasp_to(label, xy, (grasp.arm,)))
        s.place(label, grasp.arm, self.target(label, xy), text=self.noun, dest=self.dest_text(xy), **self.spec.place_kw)
        self.last_arm = grasp.arm

    def dest_text(self, xy):
        return f"the {region(xy)} of the table"

    def spots(self, label, near):
        """Free table spots, nearest to ``near`` first."""
        cand = [np.array([x, y]) for x in np.arange(-0.36, 0.37, 0.06) for y in np.arange(-0.24, 0.01, 0.06)]
        cand.sort(key=lambda c: np.linalg.norm(c - near))
        return [c for c in cand if self.free(label, c)]

    def run(self, goals, describe):
        """goals: label -> xy. ``describe(label)`` -> subtask text."""
        p = self.p
        last = None
        while True:
            pending = [lbl for lbl in goals if np.linalg.norm(self.xy(lbl) - goals[lbl]) > 0.015]
            if not pending:
                break
            progressed = False
            # 1) direct moves into free goal spots, preferring the arm used last (no handoff wait)
            for label in sorted(pending, key=lambda lbl: np.linalg.norm(self.xy(lbl) - goals[lbl])):
                if not self.free(label, goals[label]):
                    continue
                arms = arm_order(p, self.xy(label))
                if last is not None and p.queue.get(1 - last):
                    arms = (last,)
                g = self.grasp_to(label, goals[label], arms)
                if g is None:
                    continue
                if last is not None and g.arm != last:
                    self.s.home([last], background=True)
                self.move(label, goals[label], g, describe(label))
                last = g.arm
                progressed = True
                break
            if progressed:
                continue
            # 2) relay: carry to a free spot the other arm can take it from
            for label in pending:
                if not self.free(label, goals[label]):
                    continue
                src = arm_order(p, self.xy(label))[0]
                for spot in self.spots(label, (self.xy(label) + goals[label]) / 2):
                    g1 = self.grasp_to(label, spot, (src,))
                    if g1 is None:
                        continue
                    self.move(label, spot, g1, f"move the {self.noun} toward the other arm")
                    last = src
                    progressed = True
                    break
                if progressed:
                    break
            if progressed:
                continue
            # 3) clear a blocker out of some pending goal
            for label in pending:
                blockers = [
                    o
                    for o in self.labels
                    if o != label
                    and np.linalg.norm(self.xy(o) - goals[label]) < self.radius[label] + self.radius[o] + 0.012
                ]
                for b in blockers:
                    for spot in self.spots(b, self.xy(b)):
                        if any(
                            np.linalg.norm(spot - goals[lbl]) < self.radius[b] + self.radius[lbl] + 0.012
                            for lbl in goals
                            if lbl != b
                        ):
                            continue
                        g = self.grasp_to(b, spot, arm_order(p, self.xy(b)))
                        if g is None:
                            continue
                        self.move(b, spot, g, f"move the {self.noun} out of the way")
                        last = g.arm
                        progressed = True
                        break
                    if progressed:
                        break
                if progressed:
                    break
            if not progressed:
                fail("rearrangement stuck: no feasible move")
        p.drain()
        self.s.home([0, 1])


def region(xy):
    x, y = xy
    side = "left" if x < -0.12 else "right" if x > 0.12 else "center"
    return side


class DollSpec(StackSpec):
    text = "nesting doll"
    lift = 0.06

    def grasp(self, label):
        pos, _ = self.p.pose(label)
        top = self.p.top_z(label)
        h = top - self.p.bottom_z(label)
        # Head/upper body: narrow enough for the jaws on every size.
        tcp = np.r_[pos[:2], top - min(0.03, 0.3 * h)]
        return tcp, [np.deg2rad(a) for a in range(-180, 180, 45)]


@task("sort_nesting_dolls_by_size")
def sort_nesting_dolls_by_size(p, s):
    labels = [f"doll{i}" for i in range(5)]
    lm = p.lm
    # Official ordering: model_id % 5 descending (4 = smallest) from left to right.
    order = sorted(labels, key=lambda lbl: lm.get_instance_metadata(0, label=lbl)["model_id"] % 5, reverse=True)
    spec = DollSpec(p)
    r = Rearranger(p, s, spec, labels, noun="nesting doll")
    widths = [2 * r.radius[lbl] for lbl in order]
    gap = 0.018
    total = sum(widths) + gap * 4
    best = None
    for y in (-0.15, -0.12, -0.18, -0.10, -0.20):
        for x0 in (-total / 2, -total / 2 - 0.03, -total / 2 + 0.03):
            xs, x = [], x0
            for w in widths:
                xs.append(x + w / 2)
                x += w + gap
            goals = {lbl: np.array([xs[i], y]) for i, lbl in enumerate(order)}
            # Fewer objects already in the way of the row = fewer extra moves.
            conflicts = sum(
                1
                for lbl in labels
                for g in goals.values()
                if np.linalg.norm(r.xy(lbl) - g) < r.radius[lbl] + 0.05
                and np.linalg.norm(r.xy(lbl) - goals[lbl]) > 0.015
            )
            done = sum(np.linalg.norm(r.xy(lbl) - goals[lbl]) < 0.015 for lbl in labels)
            score = conflicts - 2 * done
            if best is None or score < best[0]:
                best = (score, goals)
    goals = best[1]
    size = {lbl: lm.get_instance_metadata(0, label=lbl)["model_id"] % 5 for lbl in labels}
    words = ["largest", "second largest", "middle", "second smallest", "smallest"]
    r.run(goals, lambda lbl: f"place the {words[size[lbl]]} nesting doll in its slot in the row")
    return True


def swap_T_via_buffer(p, s, poses, goal):
    """Fallback swap: park the first T on a free table spot with half its yaw change.

    Splitting the rotation over two grasps keeps each wrist turn within joint6
    range; any arm may do any move.
    """
    names = {"t0": "first T block", "t1": "second T block"}

    def grasp_for(label, target, yaw, arms, exclude=()):
        for tcp, yaws in s.mesh_grasps(label):
            g = s.pick_place_grasp(label, tcp, yaws, arms, target, place_yaw=yaw, lift=0.06, exclude=exclude)
            if g is not None:
                return g
        return None

    def carry(label, target, yaw, text, exclude=()):
        arms = arm_order(p, p.pose(label)[0])
        g = grasp_for(label, target, yaw, arms, exclude)
        if g is None:
            return None
        p.begin_subtask(text, object=label, target=[float(v) for v in target[:2]])
        p.drain()
        s.pick(
            label, g, lift=0.06, text="T-shaped block", replan=lambda: grasp_for(label, target, yaw, (g.arm,), exclude)
        )
        s.place(label, g.arm, target, target_yaw=yaw, text="T-shaped block", dest="its target spot", precise=True)
        s.home([g.arm], background=True)
        return g

    for first, second in (("t0", "t1"), ("t1", "t0")):
        pos0, q0 = poses[first]
        mid_yaw = wrap(yaw_of(q0) + wrap(yaw_of(goal[first][1]) - yaw_of(q0)) / 2)
        for bxy in ((-0.20, -0.12), (0.20, -0.12), (-0.20, -0.22), (0.20, -0.22), (0.0, -0.25)):
            buf = np.r_[bxy, pos0[2]]
            if min(np.linalg.norm(buf[:2] - poses[lbl][0][:2]) for lbl in ("t0", "t1")) < 0.10:
                continue
            if grasp_for(first, buf, mid_yaw, arm_order(p, pos0)) is None:
                continue
            if carry(first, buf, mid_yaw, f"move the {names[first]} to a free spot") is None:
                continue
            if (
                carry(
                    second,
                    goal[second][0],
                    yaw_of(goal[second][1]),
                    f"place the {names[second]} at the {names[first]}'s original pose",
                )
                is None
            ):
                fail(f"cannot place {second} after buffering {first}")
            if (
                carry(
                    first,
                    goal[first][0],
                    yaw_of(goal[first][1]),
                    f"place the {names[first]} at the {names[second]}'s original pose",
                )
                is None
            ):
                fail(f"cannot place {first} from the buffer")
            p.drain()
            s.home([0, 1])
            return True
    fail("no assignment for the swap, even via a table buffer")
