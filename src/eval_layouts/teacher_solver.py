"""Run the unchanged Panda cuRobo planner outside Isaac's CUDA context."""

import multiprocessing
import os
from types import SimpleNamespace

import numpy as np


def serve(connection, device, config):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    import torch

    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    from env.planner_manager.curobo_planner import CuroboPlanner
    from src.datagen.ik import solve_nearby

    try:
        planner = CuroboPlanner(**config)
        connection.send(("ready", planner.use_cuda_graph))
        while True:
            method, args = connection.recv()
            result = getattr(planner, method)(*args)
            if method == "solve_ik_to_joint":
                current, pose, origin = args
                if result["status"] != "Success" or np.max(np.abs(result["joint_value"] - current)) > 0.35:
                    nearby = solve_nearby(planner, current, pose, origin)
                    result = (
                        {"status": "NoContinuousIK"} if nearby is None else {"status": "Success", "joint_value": nearby}
                    )
            connection.send(("ok", result))
    except EOFError:
        os._exit(0)
    except BaseException as exc:
        try:
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
        finally:
            os._exit(1)


class TeacherSolver:
    def __init__(self, robot, dt, device):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        config = dict(
            robot_origin_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            active_joints_name=list(robot.arm_joints_name),
            all_joints=list(robot.arm_joints_name),
            dt=dt,
            yml_path=str(robot.curobo_yml_path),
            table_height=0.74 - robot.entity_origin_pose[2],
        )
        self.process = context.Process(target=serve, args=(child, device, config), daemon=True)
        self.process.start()
        child.close()
        status, result = self.connection.recv()
        if status != "ready":
            raise RuntimeError(f"Offline teacher setup failed: {result}")
        self.use_cuda_graph = result
        self.device_cfg = SimpleNamespace(device=f"cuda:0 (isolated physical GPU {device})")

    def call(self, method, *args):
        from src.datagen.planner import PlanningFailure

        self.connection.send((method, args))
        status, result = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Offline teacher failed: {result}")
        if result.get("status") == "NoContinuousIK":
            raise PlanningFailure(
                f"Support-arm contact motion lost a continuous IK solution at {np.asarray(args[1])[:3].tolist()}"
            )
        return result

    def plan_path(self, *args):
        return self.call("plan_path", *args)

    def plan_joint(self, *args):
        return self.call("plan_joint", *args)

    def solve_ik_to_joint(self, *args):
        return self.call("solve_ik_to_joint", *args)
