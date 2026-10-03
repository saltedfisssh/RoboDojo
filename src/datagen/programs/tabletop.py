"""Ordering, containment, rim grasping and sequential cover/uncover programs."""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat


def retain_swap_endpoints(p):
    """Keep the official final block positions true through button release and settling."""
    for label, mat in (("target0", "mat2"), ("target1", "mat0")):
        p.terminal_checks.append(
            p.env.reward_manager.is_pointA_close_to_pointB(label_A=label, label_B=mat, threshold=0.03)
        )


def swap_last_block(p, target):
    """Pinch the lower side faces and carry the final block above its mat."""
    from src.datagen.planner import PlanningFailure

    label = "target0"
    pos, _ = p.object_pose(label)
    arm, quat = p.pick(
        label,
        arm=1,
        grasp_point=pos + [0, 0, -0.004],
        yaw_candidates=(191, 11),
        grasp_opening=0.30,
        lift_height=0.06,
        minimum_lift=0.04,
        linear_approach=True,
        linear_lift=True,
    )
    # The pad constrains rotation during closure. Lift gently first, then
    # increase grip tension in free space; the former 0.30 carrying command
    # allowed the 250 g block to roll and eventually slide out of the fingers.
    p.stage("secure_grasp", "Secure the lifted block before moving to the right mat.", arm=arm, object_label=label)
    for opening in np.linspace(0.30, 0.12, 11)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(5)
    held, _ = p.object_pose(label)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    desired_object = np.asarray(target) + [0, 0, 0.05]
    p.stage("transfer", "Carry the held block above the right mat.", arm=arm, object_label=label)
    p.annotation["goal"].update(object_position=desired_object.tolist(), position_tolerance_m=0.015)
    p.linear(arm, desired_object + ee[:3] - held, quat, step=0.002, settle=15)
    actual, _ = p.object_pose(label)
    if np.linalg.norm(actual - desired_object) > 0.015:
        raise PlanningFailure("The final block slipped during its transfer")
    p.place(label, target, arm, quat, precise=True, contact=True, approach_clearance=0.05)
    p.home(arm)


def swap_second_block(p, *, arms=(1, 0)):
    """Carry the second block through a supported table transfer between the two arms."""
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "swap_blocks":
        raise ValueError("swap_second_supported_buffer requires swap_blocks")
    label = "target1"
    target, _ = p.object_pose("mat0")
    position, _ = p.object_pose(label)
    target[2] = p.object_vertices("mat0")[:, 2].max() + position[2] - p.object_vertices(label)[:, 2].min() + 0.002
    buffer = np.array([0.0, -0.27, 0.767 + position[2] - p.object_vertices(label)[:, 2].min()])
    for arm, destination, yaws in ((1, buffer, (191, 11, 180, 0)), (0, target, (169, 349, 180, 0))):
        if arm not in arms:
            continue
        position, _ = p.object_pose(label)
        _, quat = p.pick(
            label,
            arm=arm,
            grasp_point=position + [0, 0, -0.004],
            yaw_candidates=yaws,
            grasp_opening=0.30,
            lift_height=0.06,
            minimum_lift=0.04,
            linear_approach=True,
            linear_lift=True,
        )
        p.stage("secure_grasp", "Secure the lifted block before its supported handoff.", arm=arm, object_label=label)
        for opening in np.linspace(0.30, 0.12, 11)[1:]:
            p.gripper(arm, float(opening), frames=1)
        p.hold(5)
        # The right-hand block slipped during the last few millimeters of
        # descent. Keep the carrying squeeze and release 14 mm above the
        # table; the original buffer already includes 2 mm of clearance.
        release = destination + ([0, 0, 0.012] if arm == 1 else np.zeros(3))
        support = destination - ([0, 0, 0.002] if arm == 1 else np.zeros(3))
        p.place(
            label,
            release,
            arm,
            quat,
            target_quaternion=[1, 0, 0, 0],
            precise=True,
            contact=True,
            linear_transfer=True,
            approach_clearance=0.048 if arm == 1 else 0.06,
        )
        p.home(arm)
        actual, _ = p.object_pose(label)
        if np.linalg.norm(actual - support) > 0.012:
            raise PlanningFailure(f"The supported block handoff missed its destination: {actual - support}")
    return {"second_block_on_left_mat": 0 in arms, "second_block_at_buffer": arms[-1] == 1}


def run(planner, task):
    p = planner
    from src.datagen.assembly_layouts import TASKS as ASSEMBLY_TASKS
    from src.datagen.particle_layouts import TASKS as PARTICLE_TASKS
    from src.datagen.social_layouts import TASKS as SOCIAL_TASKS

    if task in SOCIAL_TASKS:
        from src.datagen.programs.social import run as run_social

        return run_social(p, task)
    if task in PARTICLE_TASKS:
        from src.datagen.programs.particles import run as run_particles

        return run_particles(p, task)
    if task in ASSEMBLY_TASKS:
        from src.datagen.programs.assembly import run as run_assembly

        return run_assembly(p, task)
    if task == "arrange_largest_number":
        lm = p.env.scene_manager.layout_manager
        labels = sorted(
            (f"digit_{i}" for i in range(4)),
            key=lambda label: lm.get_instance_metadata(0, label=label)["model_id"] % 10,
            reverse=True,
        )
        for i, label in enumerate(labels):
            mat, _ = p.object_pose(f"mat_{i}")
            pos, _ = p.object_pose(label)
            target = mat.copy()
            target[2] = (
                p.object_vertices(f"mat_{i}")[:, 2].max() + pos[2] - p.object_vertices(label)[:, 2].min() + 0.002
            )
            # Across the 2's diagonal, the default jaw direction can lose contact
            # during transport; pinch its upper/lower bars instead. Keep the
            # calibrated fingertip height: lowering it blocks closure on the table.
            value = lm.get_instance_metadata(0, label=label)["model_id"] % 10
            yaws = (90, 270, 180, 0) if value == 2 else (180, 90, 270, 0)
            p.relocate(label, target, target_quaternion=[1, 0, 0, 0], precise=True, yaw_candidates=yaws)
    elif task == "classify_objects":
        for i in range(3):
            label = f"cat{i}_0"
            arm, quat = p.pick(label)
            target, _ = p.object_pose(f"basket{i}")
            pos, _ = p.object_pose(label)
            target[2] = 0.778 + pos[2] - p.object_vertices(label)[:, 2].min()
            p.place(label, target, arm, quat)
            p.home(arm)
    elif task == "cover_blocks":
        labels = sorted(("red", "green", "blue"), key=lambda label: p.object_pose(label)[0][0])
        starts = {f"cup{i}": p.object_pose(f"cup{i}")[0].copy() for i in range(3)}
        assigned = {}
        for i, label in enumerate(labels):
            cup = f"cup{i}"
            target, _ = p.object_pose(label)
            target[2] = starts[cup][2]
            p.relocate(cup, target)
            assigned[label] = cup
        for label in ("red", "green", "blue"):
            cup = assigned[label]
            p.relocate(cup, starts[cup])
    elif task == "stack_bowls":
        base = "bowl0"
        for label in ("bowl1", "bowl2"):
            pos, _ = p.object_pose(label)
            arm = int(pos[0] >= 0)
            vertices = p.object_vertices(label)
            # Pinch the near-side rim, rather than trying to enclose a > 9 cm bowl.
            grasp = pos.copy()
            grasp[0] = vertices[:, 0].min() + 0.003 if arm == 0 else vertices[:, 0].max() - 0.003
            grasp[2] = vertices[:, 2].max() - 0.012
            arm, quat = p.pick(label, arm=arm, grasp_point=grasp, yaw_candidates=(180, 0), lift_height=0.10)
            target, _ = p.object_pose(base)
            target[2] += 0.028
            p.place(label, target, arm, quat)
            p.home(arm)
            base = label
    elif task == "display_letters":
        for i in range(8):
            label = f"letter{i}"
            pos, _ = p.object_pose(label)
            target = np.array([-0.28 + i * 0.08, -0.18, pos[2]])
            p.relocate(label, target, target_quaternion=euler2quat(-np.pi / 2, 0, 0))
    elif task == "fasten_screws":
        for i in range(3):
            label = f"nut{i}"
            pos, _ = p.object_pose(label)
            target, _ = p.object_pose(f"bolt{i}")
            target[2] = (
                p.object_vertices(f"bolt{i}")[:, 2].max() - 0.018 + pos[2] - p.object_vertices(label)[:, 2].min()
            )
            p.relocate(label, target, target_quaternion=[1, 0, 0, 0])
    elif task == "insert_tubes":
        for i, hole in enumerate((4, 0, 2)):
            label = f"tube{i}"
            pos, object_q = p.object_pose(label)
            # Pinch near the capped end, leaving both fingers above the rack
            # when the long +Y tube axis is turned vertical and inserted.
            grasp = pos + quat2mat(object_q) @ [0, 0.11, 0]
            arm, quat = p.pick(label, grasp_point=grasp, yaw_candidates=(180, 0))
            target = p.support_pose("slot", f"test_tube_slot/{hole}")[:3]
            target[2] = p.object_vertices("slot")[:, 2].min() + 0.003
            p.place(label, target, arm, quat, target_quaternion=euler2quat(np.pi / 2, 0, 0), precise=True, contact=True)
            p.home(arm)
    elif task == "plug_in_charger":
        from src.datagen.ik import approach_ik, plan_approach
        from src.datagen.planner import PlanningFailure

        pos, object_q = p.object_pose("charger")
        rotation = quat2mat(object_q)
        if rotation[2, 1] > 0.8:
            arm, quat = p.pick(
                "charger",
                grasp_point=pos + [0, 0, 0.038],
                yaw_candidates=(180, 0),
                lift_height=0.10,
                linear_approach=True,
            )
        else:
            # Approach the broad body from behind at a feasible tilt. Turning
            # the plug upright then keeps the wrist above the power strip.
            robot = p.actions.robots[0]
            ee = p.rm.get_real_endpose(robot)[0]
            p.stage("clear_table", "Raise the empty left gripper before grasping the charger.", arm=0)
            p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
            pos, object_q = p.object_pose("charger")
            rotation = quat2mat(object_q)
            grasp = pos + rotation @ [0, 0.030, 0.006]
            solver = p.rm.planner[robot.robot_name]
            approach_result = None
            for tilt in (20, 30, 45):
                sine, cosine = np.sin(np.deg2rad(tilt)), np.cos(np.deg2rad(tilt))
                tool = rotation @ np.array([[0, 1, 0], [-sine, 0, -cosine], [-cosine, 0, sine]])
                candidate_q = mat2quat(tool)
                wrist = grasp - tool[:, 0] * robot.gripper_bias
                joints = p.rm.get_joint(robot)[0]
                grasp_ik = solver.solve_ik_to_joint(joints, np.r_[wrist, candidate_q], robot.entity_origin_pose)
                if grasp_ik["status"] != "Success":
                    candidate = approach_ik(solver, joints, np.r_[wrist, candidate_q], robot.entity_origin_pose)
                    if candidate is None:
                        print(f"[datagen] Flat charger {tilt} deg: grasp IK rejected", flush=True)
                        continue
                approach_result = plan_approach(
                    solver, joints, np.r_[wrist - tool[:, 0] * 0.07, candidate_q], robot.entity_origin_pose
                )
                print(f"[datagen] Flat charger {tilt} deg: approach {approach_result['status']}", flush=True)
                if approach_result["status"] == "Success":
                    break
            if approach_result is None or approach_result["status"] != "Success":
                raise PlanningFailure("No collision-checked approach to the flat charger")
            arm, quat = p.pick_side("charger", grasp, tool, 0, pregrasp_opening=0.70, approach_result=approach_result)
        target = p.support_pose("socket", "socket/1")[:3]
        target[2] += 0.014 - 0.017
        p.place("charger", target, arm, quat, target_quaternion=euler2quat(np.pi / 2, 0, 0), precise=True, contact=True)
        p.home(arm)
    elif task == "deposit_coin":
        grasp, _ = p.object_pose("coin0")
        # Pinch the diameter: the coin's thickness is below the X5 jaw's useful closure.
        grasp[2] += 0.006
        arm, quat = p.pick("coin0", grasp_point=grasp, yaw_candidates=(180, 0))
        opening = p.functional_pose("piggy_bank", "slot_opening")
        target = opening[:3].copy()
        target[2] += 0.017
        data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="coin0")
        insert_frame = data["active"]["functional"]["coin_insert_center_same_side"]["frame"][0]
        target_q = mat2quat(quat2mat(opening[3:]) @ quat2mat(insert_frame[3:]).T)
        p.place("coin0", target, arm, quat, target_quaternion=target_q, precise=True, contact=True)
        p.home(arm)
    elif task == "push_T":
        from src.datagen.programs.pushing import push_t

        push_t(p)
    elif task == "press_by_number":
        from src.datagen.programs.contacts import press_button

        lm = p.env.scene_manager.layout_manager
        for i in range(2):
            count = lm.get_instance_metadata(0, label=f"num{i}")["model_id"]
            for _ in range(count):
                press_button(p, f"button{i}")
            press_button(p, "button2")
        p.home(0)
        p.home(1)
    elif task in ("build_tower", "put_bottles_into_dustbin", "pour_balls_into_vase", "pack_objects_into_box"):
        from src.datagen.programs.structures import run

        run(p, task)
    elif task == "play_Xylophone":
        from src.datagen.programs.contacts import play_xylophone

        play_xylophone(p)
    elif task == "swap_blocks":
        from src.datagen.programs.contacts import press_button

        retain_swap_endpoints(p)
        for label, mat in (("target0", "mat1"), ("target1", "mat0"), ("target0", "mat2")):
            target, _ = p.object_pose(mat)
            pos, _ = p.object_pose(label)
            target[2] = p.object_vertices(mat)[:, 2].max() + pos[2] - p.object_vertices(label)[:, 2].min() + 0.002
            # Use each arm in its own workspace. Cross-table moves use a
            # physical table buffer before the receiving hand picks up again.
            if mat == "mat2":
                swap_last_block(p, target)
            elif label == "target1":
                swap_second_block(p)
            else:
                p.relocate(label, target, target_quaternion=[1, 0, 0, 0])
            pressing_arm = press_button(p, "button0")
            # The confirmation button is immediately in front of the shared
            # transfer buffer. Park the pressing arm before the other arm
            # carries the next block through that space.
            p.home(pressing_arm)
    elif task == "play_tic_tac_toe":
        from src.datagen.programs.opponent import play_tic_tac_toe

        play_tic_tac_toe(p)
    elif task == "fill_pen_holder":
        from src.datagen.programs.containers import fill_pen_holder

        fill_pen_holder(p)
    else:
        raise ValueError(task)
