import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.datagen.annotations import segments_from_frames
from src.datagen.atomic import evaluate_atomic


def test_official_object_goal_preserves_alternatives_and_rejects_wrong_height_or_orientation():
    pytest.importorskip("shapely", reason="Run official predicate integration in the RoboDojo environment")
    from env.reward_manager.reward_manager import RewardManager
    from src.datagen.state import encode_context

    poses = {"item": [[0, 0, 1.5], [1, 0, 0, 0]], "support": [[0, 0, 0], [1, 0, 0, 0]]}
    manager = RewardManager(1)
    manager.func_parser.layout_manager = SimpleNamespace(
        get_instance_name=lambda **kwargs: kwargs["label"],
        get_instance_pose=lambda **kwargs: poses[kwargs["inst_name"]],
    )
    env = SimpleNamespace(
        reward_manager=manager,
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0, 0, 1, 1, 0, 0, 0])],
        ),
    )
    predicates = [
        manager.is_A_up_B(label_A="item", label_B="support", z_threshold_min=1, z_threshold_max=2),
        [manager.is_axis_up(label="item", axis=[1, 0, 0]), manager.is_axis_up(label="item", axis=[0, 0, 1])],
    ]
    # Exercise the persisted annotation path; plain JSON lists must not
    # change predicate tuples into alternate AND/OR groups.
    annotation = json.loads(
        json.dumps(
            {
                "primitive": "settle_in_box",
                "arm": 0,
                "object": "item",
                "goal": {
                    "official_object_checks": encode_context(predicates),
                    "ee_position": [1, 0, 1],
                    "ee_quaternion": [1, 0, 0, 0],
                },
            }
        )
    )
    result = evaluate_atomic(env, annotation)
    assert result["passed"] and result["ee_position_error_m"] == 1
    poses["item"][1] = [np.sqrt(0.5), 0, -np.sqrt(0.5), 0]
    assert evaluate_atomic(env, annotation)["passed"]
    poses["item"][1] = [0, 1, 0, 0]
    assert not evaluate_atomic(env, annotation)["passed"]
    poses["item"] = [[0, 0, 2.1], [1, 0, 0, 0]]
    assert not evaluate_atomic(env, annotation)["passed"]


def test_multi_waypoint_segment_keeps_final_goal():
    rows = [
        {"segment_id": 0, "primitive": "pour", "text": "Pour the cup.", "goal": {"ee_position": [i, 0, 1]}}
        for i in range(3)
    ]
    segment = segments_from_frames(rows)[0]
    assert segment["goal"] == {"ee_position": [2, 0, 1]}
    assert (segment["start_frame"], segment["end_frame"]) == (0, 3)


def test_successful_lift_does_not_require_the_demonstrators_wrist_pose():
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0, 0, 1, 1, 0, 0, 0])],
        ),
        scene_manager=SimpleNamespace(
            layout_manager=SimpleNamespace(get_instance_pose=lambda index, label: ([0, 0, 1], [1, 0, 0, 0]))
        ),
    )
    annotation = {
        "primitive": "lift",
        "arm": 0,
        "object": "cube",
        "goal": {"object_min_z": 0.9, "ee_position": [1, 0, 1], "ee_quaternion": [1, 0, 0, 0]},
    }
    result = evaluate_atomic(env, annotation)
    assert result["passed"]
    assert result["ee_position_error_m"] == 1
    annotation["goal"]["object_min_z"] = 1.1
    assert not evaluate_atomic(env, annotation)["passed"]


def test_precision_placement_requires_object_alignment_and_handles_quaternion_sign():
    pose = [[0, 0, 1], [-1, 0, 0, 0]]
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(robot_list=[]),
        scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(get_instance_pose=lambda index, label: pose)),
    )
    annotation = {
        "primitive": "place",
        "object": "tube",
        "goal": {
            "object_position": [0, 0, 1],
            "object_quaternion": [1, 0, 0, 0],
            "position_tolerance_m": 0.006,
            "orientation_tolerance_deg": 7,
        },
    }
    assert evaluate_atomic(env, annotation)["passed"]
    pose[1] = [np.cos(np.deg2rad(10)), 0, 0, np.sin(np.deg2rad(10))]
    assert not evaluate_atomic(env, annotation)["passed"]
    pose[:] = [[0.01, 0, 1], [1, 0, 0, 0]]
    assert not evaluate_atomic(env, annotation)["passed"]


def test_insertion_alignment_uses_object_xy_and_keeps_height_free():
    pose = [[0.0002, 0, 1.2], [1, 0, 0, 0]]
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0.2, 0, 1.3, 1, 0, 0, 0])],
        ),
        scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(get_instance_pose=lambda index, label: pose)),
    )
    annotation = {
        "primitive": "align_insertion",
        "arm": 0,
        "object": "ring",
        "goal": {
            "object_xy_position": [0, 0],
            "position_tolerance_m": 0.0003,
            "ee_position": [0, 0, 1],
            "ee_quaternion": [1, 0, 0, 0],
        },
    }
    assert evaluate_atomic(env, annotation)["passed"]
    pose[0] = [0, 0.0004, 1.2]
    assert not evaluate_atomic(env, annotation)["passed"]


def test_strike_measures_tool_contact_point_even_with_a_different_wrist_pose():
    point = [0, 0, 0.8, 1, 0, 0, 0]
    wrist = np.array([0.1, 0, 1, 1, 0, 0, 0])
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")], get_real_endpose=lambda arm: [wrist]
        ),
        scene_manager=SimpleNamespace(
            layout_manager=SimpleNamespace(
                get_instance_metadata=lambda index, label: {},
                get_instance_name=lambda index, label: label,
                get_functional_points=lambda *args, **kwargs: [point],
            )
        ),
    )
    annotation = {
        "primitive": "strike",
        "arm": 0,
        "object": "mallet",
        "goal": {
            "functional_point": {
                "label": "mallet",
                "tag": "beat",
                "position": [0, 0, 0.8],
                "position_tolerance_m": 0.01,
            },
            "ee_position": [0, 0, 1],
            "ee_quaternion": [1, 0, 0, 0],
        },
    }
    assert evaluate_atomic(env, annotation)["passed"]
    # A slipped mallet must fail even if the wrist matches the recorded pose.
    wrist[0] = 0
    point[0] = 0.03
    result = evaluate_atomic(env, annotation)
    assert result["ee_position_error_m"] == 0
    assert not result["passed"]


def test_opponent_turn_requires_queue_drained_and_successful_physical_check():
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(robot_list=[]),
        interact=True,
        datagen_task="play_tic_tac_toe",
        query_support_times=[1],
        pending_support_check=[(0, 2)],
        support_arm_action=[[{}]],
        success=[True],
    )
    annotation = {"primitive": "observe_opponent", "goal": {"opponent_turn_completed": 1}}
    assert not evaluate_atomic(env, annotation)["passed"]
    env.support_arm_action = [[]]
    assert not evaluate_atomic(env, annotation)["passed"]
    env.pending_support_check = [None]
    assert evaluate_atomic(env, annotation)["passed"]
    env.success = [False]
    assert not evaluate_atomic(env, annotation)["passed"]
    env.success = [True]
    annotation["goal"]["opponent_turn_completed"] = 2
    assert not evaluate_atomic(env, annotation)["passed"]


def test_face_up_goal_allows_yaw_but_rejects_face_down_even_at_the_right_wrist_pose():
    pose = [[0, 0, 0.8], [0, 0, 0, 1]]
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0, 0, 1, 1, 0, 0, 0])],
        ),
        scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(get_instance_pose=lambda index, label: pose)),
    )
    annotation = {
        "primitive": "declare_kong",
        "object": "tile",
        "arm": 0,
        "goal": {
            "object_axis_up": [0, 0, 1],
            "axis_tolerance_deg": 25,
            "ee_position": [0.1, 0, 1],
            "ee_quaternion": [1, 0, 0, 0],
        },
    }
    assert evaluate_atomic(env, annotation)["passed"]
    pose[1] = [0, 1, 0, 0]
    annotation["goal"]["ee_position"] = [0, 0, 1]
    result = evaluate_atomic(env, annotation)
    assert result["object_axis_error_deg"] == 180
    assert not result["passed"]


def test_cloth_lift_measures_both_layers_instead_of_a_static_root_or_the_wrist():
    vertices = np.array([[0, 0, 0.85], [0, 0.02, 0.85], [0, 0, 0.77]])
    obj = SimpleNamespace(sample_mesh_vertices=lambda: [vertices])
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0, 0, 1, 1, 0, 0, 0])],
        ),
        scene_manager=SimpleNamespace(
            layout_manager=SimpleNamespace(
                get_scene_object=lambda index, name: obj,
                get_instance_name=lambda index, label: label,
                get_instance_metadata=lambda index, label: {
                    "passive": {"functional": {"seam": {"id": [0, 1]}, "seam_below": {"id": [2]}}}
                },
            )
        ),
    )
    annotation = {
        "primitive": "lift_cloth",
        "object": "target",
        "arm": 0,
        "goal": {
            "garment_points": {"seam": {"min_z": 0.8}, "seam_below": {"min_z": 0.8}},
            "ee_position": [0, 0, 1],
            "ee_quaternion": [1, 0, 0, 0],
        },
    }
    assert not evaluate_atomic(env, annotation)["passed"]
    vertices[2, 2] = 0.84
    annotation["goal"]["ee_position"] = [0.1, 0, 1]
    result = evaluate_atomic(env, annotation)
    assert result["passed"]
    assert result["garment_points"]["seam"]["position"] == [0, 0.01, 0.85]


@pytest.mark.parametrize("material", ["fluid", "balls"])
def test_pouring_requires_contents_in_container_and_allows_different_valid_tool_poses(material):
    contained = False
    calls = []

    def measure(params):
        calls.append(params)
        return contained

    wrist = np.array([0, 0, 1, 1, 0, 0, 0], dtype=float)
    pose = [[0, 0, 1], [1, 0, 0, 0]]
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [wrist],
        ),
        scene_manager=SimpleNamespace(layout_manager=SimpleNamespace(get_instance_pose=lambda index, label: pose)),
        reward_manager=SimpleNamespace(func_parser=SimpleNamespace(is_A_fluid_in_B=measure, is_A_in_B=measure)),
    )
    fluid_params = {
        "label_A": "wine",
        "label_B": "cup",
        "label_C": "bottle",
        "percentage_threshold": 0.97,
        "C_residual_threshold": 0.15,
        "ignore_scattered": True,
        "max_ignore_ratio": 0.2,
    }
    outcome = (
        {"fluid_in_container": fluid_params}
        if material == "fluid"
        else {"contained_objects": ["ball0", "ball1"], "container": "vase"}
    )
    annotation = {
        "primitive": "pour",
        "arm": 0,
        "object": "bottle",
        "goal": {
            "ee_position": [0, 0, 1],
            "ee_quaternion": [1, 0, 0, 0],
            "object_position": [0, 0, 1],
            "object_quaternion": [1, 0, 0, 0],
            **outcome,
        },
    }
    assert not evaluate_atomic(env, annotation)["passed"]
    contained = True
    wrist[0] = 0.1
    pose[:] = [[0.1, 0, 1], [0, 1, 0, 0]]
    result = evaluate_atomic(env, annotation)
    assert result["passed"]
    assert result["object_position_error_m"] == pytest.approx(0.1)
    assert result["object_orientation_error_deg"] == 180
    assert result["ee_position_error_m"] == pytest.approx(0.1)
    if material == "fluid":
        assert calls == [{**fluid_params, "env_idx": 0}] * 2
    contained = False
    assert not evaluate_atomic(env, annotation)["passed"]


def test_legacy_pour_pose_only_does_not_claim_contents_were_poured():
    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0, 0, 1, 1, 0, 0, 0])],
        ),
        scene_manager=SimpleNamespace(
            layout_manager=SimpleNamespace(get_instance_pose=lambda index, label: ([0, 0, 1], [1, 0, 0, 0]))
        ),
    )
    result = evaluate_atomic(
        env,
        {
            "primitive": "pour",
            "arm": 0,
            "object": "bottle",
            "goal": {"object_position": [0, 0, 1], "ee_position": [0, 0, 1], "ee_quaternion": [1, 0, 0, 0]},
        },
    )
    assert result["object_position_error_m"] == 0
    assert result["ee_position_error_m"] == 0
    assert result["passed"] is None


def test_seated_egg_requires_depth_as_well_as_containment():
    height = 0.045
    in_holder = True
    calls = []

    def height_check(args):
        calls.append(args)
        return args["z_lower"] < height < args["z_upper"]

    env = SimpleNamespace(
        robot_manager=SimpleNamespace(
            robot_list=[SimpleNamespace(type="target")],
            get_real_endpose=lambda arm: [np.array([0.2, 0, 1, 1, 0, 0, 0])],
        ),
        reward_manager=SimpleNamespace(
            func_parser=SimpleNamespace(
                is_A_in_B=lambda args: in_holder,
                is_A_point_above_B_point_by_z_range=height_check,
            )
        ),
    )
    bounds = {
        "label_A": "egg",
        "label_B": "holder",
        "B_functional_point": "bottom",
        "z_lower": 0.01,
        "z_upper": 0.04,
    }
    annotation = {
        "primitive": "seat_egg",
        "object": "egg",
        "arm": 0,
        "goal": {
            "contained_objects": ["egg"],
            "container": "holder",
            "object_height_range": bounds,
            "ee_position": [0, 0, 1],
            "ee_quaternion": [1, 0, 0, 0],
        },
    }
    assert not evaluate_atomic(env, annotation)["passed"]
    height = 0.025
    result = evaluate_atomic(env, annotation)
    assert result["passed"]
    assert result["ee_position_error_m"] == 0.2
    assert calls[-1] == {**bounds, "env_idx": 0}
    height = 0.005
    assert not evaluate_atomic(env, annotation)["passed"]
    height, in_holder = 0.025, False
    assert not evaluate_atomic(env, annotation)["passed"]
