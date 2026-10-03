"""Optional adapter for an already running XPolicyLab joint-action server."""

import os
import sys
import uuid

import numpy as np

from src.datagen.layouts import ROOT


def make_xpolicylab_policy():
    """Use ROBODOJO_POLICY_URL; start the policy server separately with its own venv."""
    sys.path.insert(0, str(ROOT / "XPolicyLab"))
    from client_server.ws.model_client import WsModelClient

    from src.eval_client.eval_env import _patch_websockets_proxy_compat

    _patch_websockets_proxy_compat()
    run_id = f"atomic-{uuid.uuid4().hex}"
    client = WsModelClient(
        url=os.environ.get("ROBODOJO_POLICY_URL", "ws://127.0.0.1:9999"),
        evaluation_id=run_id,
        trial_id=run_id,
        action_case_id="robodojo_atomic_joint",
    )
    client.call(func_name="reset")

    class Policy:
        last_step = None

        def observe(self, obs):
            state = obs["observation.state"]
            packed = {}
            for i, side in enumerate(("left", "right")):
                packed[f"{side}_arm_joint_state"] = state[i * 7 : i * 7 + 6]
                packed[f"{side}_ee_joint_state"] = state[i * 7 + 6 : i * 7 + 7]
            instruction = (
                obs["current_subtask"] if os.environ.get("ROBODOJO_ATOMIC_INSTRUCTION", "1") == "1" else obs["task"]
            )
            payload = {
                "state": packed,
                "action": packed,
                "instruction": instruction,
                "current_subtask": obs["current_subtask"],
                "subtask_id": str(obs["subtask_id"]),
                "vision": {
                    ("cam_head" if k == "cam_high" else k): {"color": v, "shape": v.shape}
                    for k, v in obs["images"].items()
                },
                "env_idx": 0,
                "data_format_version": "v1.0",
                "additional_info": {"frequency": 25},
            }
            client.call(func_name="update_obs", obs=payload)
            self.last_step = obs["step"]

        def __call__(self, obs):
            if self.last_step != obs["step"]:
                self.observe(obs)
            actions = client.call(func_name="get_action")
            result = []
            for action in actions:
                if not isinstance(action, dict):
                    raise ValueError("Expected XPolicyLab action dictionaries; configure server with action_type=joint")
                result.append(
                    np.concatenate(
                        [
                            np.asarray(action[key]).reshape(-1)
                            for key in (
                                "left_arm_joint_state",
                                "left_ee_joint_state",
                                "right_arm_joint_state",
                                "right_ee_joint_state",
                            )
                        ]
                    )
                )
            return np.asarray(result, dtype=np.float32)

        def close(self):
            client.close()

    return Policy()
