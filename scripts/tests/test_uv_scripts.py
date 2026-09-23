"""Environment/launcher regression checks; no models, GPU, or downloads needed."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class UvScriptsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robodojo uv test ")
        self.addCleanup(self.temp.cleanup)
        self.tmp = Path(self.temp.name)
        self.venv = self.tmp / "policy project" / ".venv"
        (self.venv / "bin").mkdir(parents=True)
        (self.venv / "pyvenv.cfg").write_text("uv = test\n")
        self.log = self.tmp / "python.json"
        self.conda_log = self.tmp / "conda.log"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        # Fail loudly if a uv branch accidentally executes Conda.
        self.executable(self.bin / "conda", '#!/bin/bash\necho called >> "$CONDA_LOG"\nexit 91\n')
        self.executable(
            self.venv / "bin/python",
            """#!/bin/bash
if [[ "${1:-}" == *setup_policy_server.py ]]; then
  exec "$TEST_PYTHON" - "$@" <<'PY'
import json, os, sys
from pathlib import Path
Path(os.environ['PYTHON_LOG']).write_text(json.dumps({
    'args': sys.argv[1:], 'venv': os.environ.get('VIRTUAL_ENV'),
    'conda': os.environ.get('CONDA_PREFIX'), 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')
}))
raise SystemExit(int(os.environ.get('SERVER_STATUS', '0')))
PY
elif [[ "${1:-}" == -c && "${2:-}" == *get_action_dim* ]]; then
  echo 16
else
  exec "$TEST_PYTHON" "$@"
fi
""",
        )
        (self.venv / "bin/python3").symlink_to("python")
        (self.venv / "bin/activate").write_text(
            'export VIRTUAL_ENV="' + str(self.venv) + '"\nexport PATH="$VIRTUAL_ENV/bin:$PATH"\n'
        )
        self.sim_venv = self.tmp / "sim environment"
        shutil.copytree(self.venv, self.sim_venv, symlinks=True)
        self.env = os.environ.copy()
        self.env.update(
            PATH=f"{self.bin}:{os.environ['PATH']}",
            TEST_PYTHON=sys.executable,
            ROBODOJO_PYTHON=sys.executable,
            ROBODOJO_VENV=str(self.sim_venv),
            PYTHON_LOG=str(self.log),
            CONDA_LOG=str(self.conda_log),
            CONDA_PREFIX="/wrong/conda",
            OLA_SEM_WAN_PATH=str(self.tmp / "wan"),
            OLA_SEM_VLM_PATH=str(self.tmp / "vlm"),
            VIRTUAL_ENV="/wrong/venv",
            PYTHONPATH=f"{ROOT}:{ROOT / 'XPolicyLab'}",
        )

    @staticmethod
    def executable(path, text):
        path.write_text(text)
        path.chmod(0o755)

    def run_shell(self, args, expected=0, **extra):
        env = self.env | extra
        result = subprocess.run(
            ["bash", *map(str, args)],
            cwd=self.tmp,
            env=env,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.assertFalse(self.conda_log.exists(), "uv branch invoked Conda")
        return result

    def test_environment_isolation_and_project_paths(self):
        helper = ROOT / "scripts/internal/uv_env.sh"
        for path in [self.venv, self.venv.parent]:
            result = self.run_shell(
                [
                    "-c",
                    'source "$1"; robodojo_use_uv "$2"; '
                    'printf "%s|%s|%s" "$VIRTUAL_ENV" "${CONDA_PREFIX-unset}" "$(command -v python)"',
                    "test",
                    helper,
                    path,
                ]
            )
            self.assertEqual(result.stdout, f"{self.venv}|unset|{self.venv}/bin/python")

    def test_missing_venv_never_falls_back(self):
        self.run_shell(
            [
                "-c",
                'source "$1"; robodojo_use_uv "$2"',
                "test",
                ROOT / "scripts/internal/uv_env.sh",
                self.tmp / "missing",
            ],
            expected=1,
        )

    def test_help_works_without_an_environment(self):
        for script in [
            "robodojo-uv.sh",
            "eval_policy-uv.sh",
            "init_assets-uv.sh",
            "internal/verify_install-uv.sh",
            "RoboDojo/download_ckpt-uv.sh",
        ]:
            self.run_shell([ROOT / "scripts" / script, "--help"], ROBODOJO_VENV="/missing")

    def test_simulator_path_reaches_eval_orchestrator(self):
        result = self.run_shell(
            [
                ROOT / "scripts/robodojo-uv.sh",
                "eval",
                "--policy-dir",
                "XPolicyLab/policy/demo_policy",
                "--task",
                "stack_bowls",
                "--ckpt",
                "demo",
                "--policy-env",
                str(self.venv),
                "--dry-run",
            ]
        )
        self.assertIn("run_policy_eval.sh", result.stdout)
        command = shlex.split(result.stdout.split("dry-run: ", 1)[1].strip())
        self.assertEqual(command[-2:], [str(self.venv), str(self.sim_venv)])

    def test_policy_servers_keep_arguments_and_select_uv(self):
        for name in [
            "demo_policy",
            "ACT",
            "Pi_0",
            "GR00T_N17",
            "Spirit_v15",
            "MolmoAct2",
            "Xiaomi_Robotics_1",
            "OLA_SEM",
        ]:
            with self.subTest(policy=name):
                self.run_shell(
                    [
                        ROOT / "scripts/robodojo-uv.sh",
                        "server",
                        "--policy-dir",
                        f"XPolicyLab/policy/{name}",
                        "--task",
                        "stack_bowls",
                        "--ckpt",
                        "checkpoint with spaces",
                        "--policy-env",
                        self.venv,
                        "--policy-port",
                        "19999",
                        "--bind-host",
                        "127.0.0.1",
                        "--policy-gpu",
                        "2",
                    ]
                )
                payload = json.loads(self.log.read_text())
                self.assertEqual(payload["venv"], str(self.venv))
                self.assertEqual(payload["conda"], None)
                self.assertEqual(payload["gpu"], "2")
                self.assertIn("host=127.0.0.1", payload["args"])
                self.assertIn("port=19999", payload["args"])
                self.assertIn("ckpt_name=checkpoint with spaces", payload["args"])

    def test_server_exit_status_propagates(self):
        self.run_shell(
            [
                ROOT / "scripts/robodojo-uv.sh",
                "server",
                "--policy-dir",
                "XPolicyLab/policy/demo_policy",
                "--task",
                "stack_bowls",
                "--ckpt",
                "demo",
                "--policy-env",
                self.venv,
                "--policy-port",
                "19999",
            ],
            expected=23,
            SERVER_STATUS="23",
        )

    def test_shared_sim_client_activates_before_yaml_read(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("PyYAML is needed for the client's deploy metadata")
        scripts = self.tmp / "scripts"
        scripts.mkdir()
        self.executable(
            scripts / "eval_policy.sh",
            """#!/bin/bash
printf '%s\n' "$VIRTUAL_ENV" "${CONDA_PREFIX-unset}" "$@"
""",
        )
        result = self.run_shell(
            [
                ROOT / "XPolicyLab/utils/setup_env_client.sh",
                ROOT / "XPolicyLab/utils",
                ROOT / "XPolicyLab/policy/demo_policy/deploy.yml",
                self.venv,
                "19999",
                "RoboDojo",
                "stack_bowls",
                "arx_x5",
                "demo_policy",
                "ckpt_name=demo",
                self.tmp,
                "0",
                "2",
                "127.0.0.1",
            ],
            EVAL_ENV_TYPE="sim",
        )
        self.assertIn(str(self.venv) + "\nunset\n", result.stdout)
        self.assertIn("--eval_batch\n", result.stdout)
        self.assertIn("--device_id\n2\n", result.stdout)

    def test_git_downloaders_do_not_activate_conda(self):
        repo = self.tmp / "download repo"
        (repo / "scripts/RoboDojo").mkdir(parents=True)
        for name in [
            "init_assets.sh",
            "init_assets-uv.sh",
            "RoboDojo/download_data.sh",
            "RoboDojo/download_data-uv.sh",
        ]:
            shutil.copy2(ROOT / "scripts" / name, repo / "scripts" / name)
        for folder in ["Robots", "Object", "Material", "Eval_Layout"]:
            (repo / "Assets" / folder).mkdir(parents=True)
        (repo / "data/demo").mkdir(parents=True)
        (repo / "data/demo/.download_complete").touch()
        self.executable(self.bin / "git", '#!/bin/bash\n[[ "$*" == "lfs version" ]] || exit 97\n')
        self.run_shell([repo / "scripts/init_assets-uv.sh"], VIRTUAL_ENV="", ROBODOJO_VENV="/missing")
        self.run_shell(
            [repo / "scripts/RoboDojo/download_data-uv.sh", "huggingface", "demo"],
            VIRTUAL_ENV="",
            ROBODOJO_VENV="/missing",
        )

    def test_conda_name_keeps_legacy_activation(self):
        base = self.tmp / "conda base"
        (base / "etc/profile.d").mkdir(parents=True)
        (base / "etc/profile.d/conda.sh").write_text(
            'conda() { [[ "$1" == activate ]] || return 8; export CONDA_PREFIX="/fake/$2"; }\n'
        )
        self.executable(self.bin / "conda", '#!/bin/bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n')
        result = self.run_shell(
            [
                "-c",
                'source "$1"; xpl_activate_env legacy "$2"; printf "%s|%s" "$CONDA_PREFIX" "${VIRTUAL_ENV-unset}"',
                "test",
                ROOT / "XPolicyLab/utils/activate_env.sh",
                self.tmp,
            ],
            CONDA_EXE=str(self.bin / "conda"),
            FAKE_CONDA_BASE=str(base),
        )
        self.assertEqual(result.stdout, "/fake/legacy|unset")


if __name__ == "__main__":
    unittest.main()
