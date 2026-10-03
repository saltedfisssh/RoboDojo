"""Replayable opponent control state for the official tic-tac-toe scripts."""

from copy import deepcopy


def capture(env):
    if not getattr(env, "interact", False):
        return None
    if env.datagen_task == "imitate_sorting_sequence":
        return {
            "task": env.datagen_task,
            "query_support_times": deepcopy(env.query_support_times),
            "support_checked": deepcopy(env.support_checked),
            "next_controls": deepcopy(env.support_arm_action[0][:10]),
        }
    if env.datagen_task == "make_kong":
        return {
            "task": env.datagen_task,
            "query_support_times": deepcopy(env.query_support_times),
            "support_checked": deepcopy(env.support_checked),
            "remaining_controls": len(env.support_arm_action[0]),
        }
    if env.datagen_task != "play_tic_tac_toe":
        raise ValueError(f"Interactive checkpoint adapter not implemented for {env.datagen_task}")
    return {
        "task": env.datagen_task,
        "query_support_times": deepcopy(env.query_support_times),
        "pending_support_check": deepcopy(env.pending_support_check),
        "remaining_controls": len(env.support_arm_action[0]),
    }


def restore(env, context):
    if context is None:
        if getattr(env, "interact", False):
            raise ValueError("Interactive task checkpoint is missing opponent state")
        return
    if context["task"] != env.datagen_task:
        raise ValueError("Unsupported interactive task checkpoint")
    if env.datagen_task == "imitate_sorting_sequence":
        env.query_support_times = deepcopy(context["query_support_times"])
        env.support_checked = deepcopy(context["support_checked"])
        env.support_arm_action = [deepcopy(context["next_controls"])]
        env.datagen_support_replay = True
        return
    if env.datagen_task == "make_kong":
        env.query_support_times = deepcopy(context["query_support_times"])
        env.support_checked = deepcopy(context["support_checked"])
        env.support_arm_action = [[]]
        remaining = context["remaining_controls"]
        if remaining:
            manager = env.robot_manager
            arm = manager.plan_ee(0, "support_arm0", result=env.traj["arm"][0], need_plan=False)
            grip = manager.plan_endeffector_joint(0, "support_arm0", result=env.traj["eef"][0], need_plan=False)
            controls = [{**a, **g} for a, g in zip(arm, grip, strict=True)]
            if remaining < 0 or remaining > len(controls):
                raise ValueError("Invalid support-arm trajectory cursor")
            env.support_arm_action[0] = controls[-remaining:]
        return
    if env.datagen_task != "play_tic_tac_toe":
        raise ValueError("Unsupported interactive task checkpoint")
    env.query_support_times = deepcopy(context["query_support_times"])
    env.pending_support_check = deepcopy(context["pending_support_check"])
    env.support_arm_action = [[]]
    remaining = context["remaining_controls"]
    if remaining:
        pending = env.pending_support_check[0]
        if pending is None or remaining < 0:
            raise ValueError("Invalid opponent trajectory cursor")
        piece, cell = pending
        trajectory = env.traj[piece][cell]
        manager = env.robot_manager
        arm = manager.plan_ee(0, "support_arm0", result=trajectory["arm"], need_plan=False)
        gripper = manager.plan_endeffector_joint(0, "support_arm0", result=trajectory["eef"], need_plan=False)
        controls = [{**a, **g} for a, g in zip(arm, gripper, strict=True)]
        if remaining > len(controls):
            raise ValueError("Opponent cursor exceeds the recorded trajectory length")
        env.support_arm_action[0] = controls[-remaining:]


def turn_metrics(env, minimum_turn):
    """Require the opponent's physics check as well as its command queue ending."""
    if not getattr(env, "interact", False):
        return {"passed": False, "reason": "No interactive opponent in this task"}
    queried = int(env.query_support_times[0])
    pending = len(env.support_arm_action[0])
    if env.datagen_task == "play_tic_tac_toe":
        checked = env.pending_support_check[0] is None
    elif env.datagen_task in ("make_kong", "imitate_sorting_sequence"):
        checked = bool(env.support_checked[0])
    else:
        raise ValueError(f"Unsupported opponent metrics for {env.datagen_task}")
    stable = bool(all(env.success))
    return {
        "passed": queried >= minimum_turn and pending == 0 and checked and stable,
        "query_support_times": queried,
        "remaining_controls": pending,
        "physical_check_complete": checked,
        "environment_success": stable,
    }
