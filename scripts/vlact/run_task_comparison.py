"""Run a sequential paired sweep against already verified local policy servers."""

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("plug_in_charger", "insert_tubes", "pour_balls_into_vase")


def now():
    return datetime.now(UTC).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_file = output / "sweep_status.json"
    if status_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing sweep: {status_file}")
    files = [
        ROOT / p
        for p in (
            "src/eval_client/manipulation_oracle.py",
            "src/eval_client/subtask_oracle.py",
            "src/eval_client/eval_env.py",
            "scripts/vlact/run_task.sh",
            "scripts/vlact/run_task_comparison.py",
            "XPolicyLab/policy/starVLA/model.py",
        )
    ]
    status = dict(
        started_at=now(),
        state="running",
        seed=0,
        eval_num=1,
        runs=[],
        code_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    )

    def save():
        temp = status_file.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_file)

    save()
    for task in TASKS:
        for step, port in ((30000, 6000), (40000, 6001)):
            label = f"vlact_{step}"
            run_id = f"{label}_{task}_{output.name}"
            result_dir = (
                ROOT
                / "eval_result/RoboDojo"
                / task
                / "starVLA/arx_x5"
                / f"0_ckpt_name={label},action_type=joint,conditioning=state_oracle"
                / run_id
            )
            if result_dir.exists():
                raise FileExistsError(result_dir)
            log_path = output / f"{task}_{step}.log"
            command = ["bash", "scripts/vlact/run_task.sh", task, label, str(port), "1"]
            run = dict(
                task=task,
                checkpoint_step=step,
                port=port,
                started_at=now(),
                state="running",
                log=str(log_path),
                result_directory=str(result_dir),
                command=command,
            )
            status["runs"].append(run)
            save()
            env = os.environ.copy()
            env.update(
                ROBODOJO_RUN_ID=run_id,
                OMNI_KIT_ACCEPT_EULA="YES",
                VLACT_LAYOUT_SEED="0",
                STARVLA_ROBODOJO_NUM_ENVS="1",
                VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json",
            )
            with log_path.open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                run["pid"] = process.pid
                save()
                rc = process.wait()
            run.update(exit_code=rc, finished_at=now())
            result_path = result_dir / "_result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            valid = (
                rc == 0
                and result is not None
                and result.get("eval_time", 0) == 1
                and len(result.get("details", {})) == 1
                and all(d.get("layout_id") == 0 for d in result["details"].values())
            )
            run.update(state="complete" if valid else "invalid", result=result)
            save()
            if not valid:
                status.update(state="failed", finished_at=now())
                save()
                raise RuntimeError(f"Simulation failed acceptance: {log_path}")
    status.update(state="complete", finished_at=now())
    save()


if __name__ == "__main__":
    main()
