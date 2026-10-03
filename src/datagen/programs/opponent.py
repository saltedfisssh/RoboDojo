"""Player actions while the official support-arm opponent plays its own turns."""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat


def play_tic_tac_toe(p):
    from src.datagen.planner import PlanningFailure

    for i in range(5):
        label = f"player_piece{i}"
        _, empty = p.env.check_chessboard_state(0)
        if not empty:
            raise PlanningFailure("The opponent left no empty cell for the player")
        arm, quat = p.pick(label)
        place_held_tic_piece(p, label, arm, quat)
        if i < 4:
            p.stage("observe_opponent", "Wait and observe the opponent place its next piece.")
            p.annotation["goal"]["opponent_turn_completed"] = i + 1
            for _ in range(600):
                p.hold(1)
                if not p.env.success[0]:
                    raise PlanningFailure("The official opponent's physical move failed")
                if p.env.query_support_times[0] >= i + 1 and not p.env.support_arm_action[0]:
                    break
            else:
                raise PlanningFailure("Timed out waiting for the opponent")


def place_held_tic_piece(p, label, arm, quat):
    """Place a held piece, transferring on the table if its arm cannot reach a free cell."""
    from src.datagen.planner import PlanningFailure

    for attempt in range(2):
        _, empty = p.env.check_chessboard_state(0)
        if not empty:
            raise PlanningFailure("The opponent left no empty cell for the player")
        robot = p.actions.robots[arm]
        solver = p.rm.planner[robot.robot_name]
        lifted, object_q = p.object_pose(label)
        ee = p.rm.get_real_endpose(robot)[0]
        data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
        vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])
        for cell in sorted(
            empty, key=lambda j: np.linalg.norm(p.support_pose("checkerboard", f"cell/{j}")[:2] - lifted[:2])
        ):
            target = p.support_pose("checkerboard", f"cell/{cell}")[:3]
            # Lean the thin piece forward for the far row, then let it settle flat.
            target_q = euler2quat(np.pi / 6 if target[1] > -0.03 else 0, 0, 0)
            rotation = quat2mat(target_q)
            target[2] += -(vertices @ rotation.T)[:, 2].min() + 0.003
            delta = rotation @ quat2mat(object_q).T
            wrist = target + delta @ (ee[:3] - lifted)
            wrist_q = mat2quat(delta @ quat2mat(ee[3:]))
            above = wrist + [0, 0, 0.04]
            if all(
                solver.solve_ik_to_joint(p.rm.get_joint(robot)[0], np.r_[point, wrist_q], robot.entity_origin_pose)[
                    "status"
                ]
                == "Success"
                for point in (above, wrist)
            ):
                break
        else:
            if attempt:
                raise PlanningFailure("Neither arm has a reachable free-cell placement after the physical transfer")
            # A randomized opponent can leave the opposite middle cell last.
            # Use the clear area in front of the board as a physical relay.
            buffer = np.array([0.0, -0.27, 0.767 + lifted[2] - p.object_vertices(label)[:, 2].min()])
            p.place(
                label,
                buffer,
                arm,
                quat,
                target_quaternion=[1, 0, 0, 0],
                precise=True,
                contact=True,
                linear_transfer=True,
                approach_clearance=0.04,
            )
            p.home(arm)
            arm, quat = p.pick(label, arm=1 - arm, linear_approach=True, linear_lift=True)
            continue
        p.stage("lower_carry", "Lower the held piece before carrying it over the board.", arm=arm, object_label=label)
        lowered = ee[:3].copy()
        lowered[2] = above[2]
        p.linear(arm, lowered, quat)
        p.place(label, target, arm, quat, target_quaternion=target_q, contact=True, approach_clearance=0.04)
        p.home(arm)
        return {"cell": int(cell), "placing_arm": arm, "used_table_transfer": bool(attempt)}
    raise PlanningFailure("No placement completed")
