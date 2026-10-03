"""Compare separate and combined tiled RGB capture with the same physics rollout."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import traceback


class CombinedRGB:
    """One tiled render product for the three cameras of a single environment."""

    def __init__(self, camera_manager):
        import warp as wp

        from env.camera_manager.capture.camera_view import CameraView

        cameras = camera_manager.cameras[0]
        resolutions = {tuple(camera._resolution) for camera in cameras}
        if len(resolutions) != 1 or camera_manager.num_envs != 1:
            raise ValueError("Combined capture requires one environment and equal camera resolutions")
        width, height = next(iter(resolutions))
        paths = [camera.prim_path for camera in cameras]
        self.view = CameraView(
            paths,
            camera_resolution=(width, height),
            output_annotators=["rgb"],
            reset_xform_properties=False,
            name="datagen_combined_cameras",
        )
        self.indices = [self.view.prim_paths.index(path) for path in paths]
        self.buffer = wp.zeros((len(cameras), height, width, 4), dtype=wp.uint8, device="cuda:0")

    def step(self, env_ids=None, cam_ids=None):
        if env_ids is not None and list(env_ids) != [0]:
            raise ValueError("Only environment zero is supported")
        array, info = self.view.get_data("rgb", out=self.buffer)
        frames = array.numpy()
        indices = range(len(self.indices)) if cam_ids is None else cam_ids
        return [{"rgb": [{"data": frames[self.indices[i]], "info": info}]} for i in indices]

    def destroy(self):
        disconnect(self.view)


def disconnect(view):
    for annotator in view._annotators.values():
        annotator.detach([view._render_product.path])
    view._render_product.destroy()


def main():
    from src.datagen.frozen import install

    revision = install()
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.kit_args = (args.kit_args or "") + " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
    args.kit_args += " --/plugins/carb.tasking.plugin/threadCount=8 --/plugins/omni.tbb.globalcontrol/maxThreadCount=8"
    from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
    from src.datagen.lease import acquire

    add_zero_delay_kit_args(args)
    lease = acquire()
    app = AppLauncher(args).app
    env = None
    report = {"source_snapshot": revision, "passed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        from PIL import Image

        from src.datagen.layouts import sample_layout
        from src.datagen.runtime import ActionAdapter, create_env
        from src.datagen.state import RigidStateAdapter

        env = create_env(app, "stack_blocks", 10000, sample_layout("stack_blocks", 10000), planning=False)
        actions = ActionAdapter(env)
        states = RigidStateAdapter(env, actions)
        snapshot = states.capture()
        command = actions.command.copy()
        sequences = {}
        for mode in ("separate", "combined"):
            if mode == "combined":
                original = env.capture_manager
                for view in original.tiled_cameras:
                    disconnect(view)
                original.tiled_render_products.clear()
                original.tiled_cameras.clear()
                combined = CombinedRGB(env.camera_manager)
                env.capture_manager = env.obs_manager.capture_manager = combined
                env.datagen_render_warmup = True
            states.restore(*snapshot)
            actions.images()  # Exclude shader/material initialization from steady-state timing.
            images, captures, physics = [], [], []
            for i in range(args.frames):
                stamp = perf_counter()
                images.append(actions.images())
                captures.append(perf_counter() - stamp)
                next_action = command.copy()
                next_action[0] += 0.04 * (i + 1) / args.frames
                stamp = perf_counter()
                actions.apply(next_action)
                physics.append(perf_counter() - stamp)
            sequences[mode] = images
            report[mode] = {
                "capture_median_s": float(np.median(captures)),
                "capture_mean_s": float(np.mean(captures)),
                "physics_mean_s": float(np.mean(physics)),
            }
            print(f"[camera_probe] {mode}: {json.dumps(report[mode])}", flush=True)
        report["mean_absolute_pixel_error"] = {
            key: float(
                np.mean(
                    [
                        np.abs(a[key].astype(float) - b[key]).mean()
                        for a, b in zip(sequences["separate"], sequences["combined"], strict=True)
                    ]
                )
            )
            for key in sequences["separate"][0]
        }
        report["speedup"] = report["separate"]["capture_mean_s"] / report["combined"]["capture_mean_s"]
        report["action_timings_seconds"] = actions.timings
        for mode, frames in sequences.items():
            for i in (0, args.frames - 1):
                canvas = np.concatenate(list(frames[i].values()), axis=1)
                Image.fromarray(canvas).save(args.report.with_name(f"camera_probe_{mode}_{i}.jpg"))
        report["passed"] = all(value < 10 for value in report["mean_absolute_pixel_error"].values())
        report["scope"] = "Camera ordering and short moving-arm rollout; inspect comparison images before deployment."
    except Exception as exc:
        traceback.print_exc()
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if env is not None:
            env.close()
        app.close()
        lease.close()


if __name__ == "__main__":
    main()
