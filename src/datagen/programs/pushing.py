"""Measured planar pushes, including re-contact from a new side for lateral correction."""

import numpy as np
from transforms3d.euler import quat2euler
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.programs.assembly import top_rotation


def stroke(p, contact_xy, delta_xy, description, stop_when, *, rotation_only=False):
    arm = 0
    quat = mat2quat(top_rotation(arm))
    # The collision mesh extends 12.57 mm beyond gripper_bias: keeping the
    # reference 2 mm above the block leaves its fingertips inside the block's
    # height interval, with clearance above the table.
    contact_z = float(p.object_vertices("t")[:, 2].max()) + 0.002
    wrist = np.r_[contact_xy, contact_z + p.actions.robots[arm].gripper_bias]
    p.stage("approach_push", "Reposition the fingers beside the T block.", arm=arm, object_label="t")
    p.move(arm, wrist + [0, 0, 0.065], quat)
    p.gripper(arm, 0, frames=6)
    p.linear(arm, wrist, quat, step=0.003)
    p.stage("straighten" if rotation_only else "push", description, arm=arm, object_label="t")
    target, target_q = p.object_pose("target_t")
    p.annotation["goal"].update(object_quaternion=target_q.tolist(), orientation_tolerance_deg=7.0)
    if not rotation_only:
        p.annotation["goal"].update(object_xy_position=target[:2].tolist(), position_tolerance_m=0.007)
    p.linear(arm, wrist + np.r_[delta_xy, 0], quat, step=0.0015, settle=3, stop_when=stop_when)
    p.stage("retract", "Lift the fingers before approaching from another side.", arm=arm, object_label="t")
    actual = p.rm.get_real_endpose(p.actions.robots[arm])[0][:3]
    p.linear(arm, actual + [0, 0, 0.065], quat, step=0.004, settle=2)


def footprint(pos, quat):
    # Exact outline of the shipped t/00000 collision mesh. Its stem runs along
    # +X. The overall bounding box's +/-Y edges are empty at the stem center.
    outline = np.array(
        [
            [-0.04, -0.03, 0],
            [-0.02, -0.03, 0],
            [-0.02, -0.01, 0],
            [0.04, -0.01, 0],
            [0.04, 0.01, 0],
            [-0.02, 0.01, 0],
            [-0.02, 0.03, 0],
            [-0.04, 0.03, 0],
        ]
    )
    return (outline @ quat2mat(quat).T + pos)[:, :2]


def section(points, axis, transverse):
    crossings = []
    for a, b in zip(points, np.roll(points, -1, axis=0), strict=True):
        den = b[1 - axis] - a[1 - axis]
        if abs(den) < 1e-9:
            continue
        alpha = (transverse - a[1 - axis]) / den
        if 0 <= alpha <= 1:
            crossings.append(float(a[axis] + alpha * (b[axis] - a[axis])))
    if not crossings:
        raise ValueError("Push line does not intersect the T collision footprint")
    return min(crossings), max(crossings)


def push_t(p):
    from src.datagen.planner import PlanningFailure

    target, target_q = p.object_pose("target_t")
    target_yaw = quat2euler(target_q)[2]

    def yaw_error():
        yaw = quat2euler(p.object_pose("t")[1])[2] - target_yaw
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    previous = None
    misses = 0
    for _ in range(12):
        pos, q = p.object_pose("t")
        if previous is not None:
            misses = misses + 1 if np.linalg.norm(pos - previous) < 0.0003 else 0
            if misses >= 2:
                raise PlanningFailure("Two consecutive T-block pushes missed physical contact")
        previous = pos.copy()
        yaw = yaw_error()
        error = target[:2] - pos[:2]
        print(
            f"[push] position={pos[:2].tolist()} error={error.tolist()} yaw_error_deg={np.rad2deg(yaw):.3f}", flush=True
        )
        if np.linalg.norm(error) < 0.0045 and abs(yaw) <= np.deg2rad(5):
            break
        points = footprint(pos, q)
        center = pos + quat2mat(q) @ [-0.01, 0, 0]
        if abs(yaw) > np.deg2rad(5):
            sign = np.sign(yaw)
            x = points[:, 0].max() - 0.006 if sign > 0 else points[:, 0].min() + 0.006
            point = np.array([x, section(points, 1, x)[1] + 0.013])
            stroke(
                p,
                point,
                [0, -0.050],
                "Straighten the T block with an offset contact on its crossbar.",
                lambda: sign * yaw_error() <= np.deg2rad(2),
                rotation_only=True,
            )
        else:
            axis = 1 if abs(error[1]) > 0.003 else 0
            direction = np.sign(error[axis])
            point = center[:2].copy()
            lower, upper = section(points, axis, point[1 - axis])
            point[axis] = lower - 0.013 if direction > 0 else upper + 0.013
            delta = np.zeros(2)
            delta[axis] = error[axis] + direction * 0.035
            stroke(
                p,
                point,
                delta,
                "Push the T block toward the target pad along the table.",
                lambda: (
                    direction * (target[axis] - p.object_pose("t")[0][axis]) <= 0.0025
                    or abs(yaw_error()) > np.deg2rad(5)
                ),
            )
    p.stage("release", "Open the fingers after aligning the T block.", arm=0, object_label="t")
    p.gripper(0, 1, frames=6)
    p.home(0)
