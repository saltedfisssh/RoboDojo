"""Check real held-out observations through both StarVLA and RoboDojo WebSockets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlact-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend-port", type=int, required=True)
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [
        str(root),
        str(root / "XPolicyLab"),
        str(args.vlact_root),
        str(root / "XPolicyLab/policy/starVLA/source_starvla"),
    ]
    from client_server.ws.model_client import WsModelClient
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    from examples.RoboDojo.data import RoboDojoDataset
    import numpy as np

    from XPolicyLab.policy.starVLA.runtime_config import validate_server_runtime_contract
    from XPolicyLab.utils.process_data import get_robot_action_dim_info, pack_robot_state

    if args.output.exists():
        raise FileExistsError(args.output)
    dataset = RoboDojoDataset(args.cache, split="val")
    direct = WebsocketClientPolicy("127.0.0.1", args.backend_port)
    metadata = direct.get_server_metadata()
    if Path(metadata["ckpt_path"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Server is using a different checkpoint")
    validate_server_runtime_contract(
        metadata, include_state=False, action_dim=14, unnorm_key="arx_x5", expected_framework="QwenOFT"
    )
    dimensions = get_robot_action_dim_info("arx_x5")
    seen, records = set(), []
    base = 0
    with WsModelClient(
        url=f"ws://127.0.0.1:{args.port}",
        evaluation_id="vlact-offline-probe",
        trial_id="validation",
        request_timeout_s=120,
    ) as adapter:
        for episode in dataset.episodes:
            index = base + episode["length"] // 2
            base += episode["length"]
            task = episode["task_index"]
            if task in seen:
                continue
            seen.add(task)
            example = dataset[index]
            images = [np.asarray(im) for im in example["image"]]
            started = time.monotonic()
            response = direct.predict_action(
                {"examples": [{"image": images, "lang": example["lang"]}], "unnorm_key": "arx_x5", "do_sample": False}
            )
            if not response.get("ok"):
                raise RuntimeError(response)
            chunk = np.asarray(response["data"]["actions"])[0]
            if chunk.shape != (50, 14) or not np.isfinite(chunk).all():
                raise ValueError("Invalid server action chunk")
            backend_seconds = time.monotonic() - started
            observation = {
                "env_idx": 0,
                "instruction": episode["task"],
                "current_subtask": example["current_subtask"],
                "subtask_id": 1,
                "vision": {
                    camera: {"color": im}
                    for camera, im in zip(("cam_head", "cam_left_wrist", "cam_right_wrist"), images)
                },
            }
            adapter.call("reset")
            adapter.call("update_obs", observation)
            max_error = 0.0
            for step in range(3):
                actions = adapter.call("get_action")
                vector = pack_robot_state({"state": actions[0]}, "joint", dimensions)
                np.testing.assert_allclose(vector, chunk[step], rtol=1e-5, atol=1e-5)
                max_error = max(max_error, float(np.max(np.abs(vector - chunk[step]))))
            # Same instruction, new atomic step: action index must restart at zero.
            adapter.call("update_obs", {**observation, "subtask_id": 2})
            reset_action = adapter.call("get_action")[0]
            vector = pack_robot_state({"state": reset_action}, "joint", dimensions)
            np.testing.assert_allclose(vector, chunk[0], rtol=1e-5, atol=1e-5)
            records.append(
                {
                    "task_index": task,
                    "task": episode["task"],
                    "episode": example["original_episode_index"],
                    "frame": example["frame_index"],
                    "backend_seconds": backend_seconds,
                    "max_adapter_error": max_error,
                    "subtask_identity_reset_passed": True,
                    "first_action": chunk[0].tolist(),
                }
            )
            print(f"verified task {task}: {len(records)} tasks", flush=True)
    direct.close()
    args.output.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "tasks": len(records),
                "simulator_evaluation": False,
                "include_state": False,
                "chunk_shape": [50, 14],
                "all_actions_finite": True,
                "adapter_action_order_verified": True,
                "subtask_identity_reset_verified": True,
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
