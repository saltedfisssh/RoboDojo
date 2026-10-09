"""Exercise relocated launchers and installer safeguards without downloads or a GPU."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class DatagenEnvironmentsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="robodojo independent clone ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "my clone"
        self.root.mkdir()
        for name in (
            "generate_data.sh",
            "datagen-uv.sh",
            "run-lerobot.sh",
            "install-lerobot-uv.sh",
            "requirements-lerobot.txt",
            "internal/uv_env.sh",
        ):
            target = self.root / "scripts" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "scripts" / name, target)
        self.env = os.environ.copy()
        for key in ("ROBODOJO_ROOT", "ROBODOJO_VENV", "LEROBOT_VENV", "LEROBOT_PYTHON"):
            self.env.pop(key, None)
        self.env.update(
            TEST_PYTHON=sys.executable,
            VIRTUAL_ENV="/unrelated/active/env",
            CONDA_PREFIX="/unrelated/conda",
            PYTHONPATH="/old/export/overlay",
        )
        self.make_env(".venv")
        self.make_env(".venv-lerobot")

    def make_env(self, name):
        path = self.root / name
        (path / "bin").mkdir(parents=True)
        (path / "pyvenv.cfg").write_text("test = true\n")
        python = path / "bin/python"
        python.write_text(
            """#!/bin/bash
exec "$TEST_PYTHON" - "$@" <<'PY'
import json, os, sys
print(json.dumps({
    'args': sys.argv[1:], 'cwd': os.getcwd(), 'venv': os.environ.get('VIRTUAL_ENV'),
    'conda': os.environ.get('CONDA_PREFIX'), 'pythonpath': os.environ.get('PYTHONPATH'),
    'pythonhome': os.environ.get('PYTHONHOME')
}))
raise SystemExit(int(os.environ.get('TEST_STATUS', '0')))
PY
"""
        )
        python.chmod(0o755)
        return path

    def run_script(self, name, *args, expected=0, **env):
        result = subprocess.run(
            ["bash", str(self.root / "scripts" / name), *args],
            cwd=self.root.parent,
            env=self.env | env,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_export_selects_own_environment_from_other_directory(self):
        # No simulator installation is needed to export, including via the legacy alias.
        shutil.rmtree(self.root / ".venv")
        for script, args, module in (
            ("generate_data.sh", ["export", "--output", "data/with spaces"], "src.expert"),
            ("generate_data.sh", ["legacy", "export", "--help"], "src.datagen.export"),
            ("datagen-uv.sh", ["review", "--help"], "src.datagen.review"),
        ):
            with self.subTest(script=script, args=args):
                data = json.loads(self.run_script(script, *args, PYTHONHOME="/wrong/python").stdout)
                self.assertEqual(data["venv"], str(self.root / ".venv-lerobot"))
                self.assertEqual(data["cwd"], str(self.root))
                self.assertEqual(data["pythonpath"], str(self.root))
                self.assertIsNone(data["conda"])
                self.assertIsNone(data["pythonhome"])
                self.assertEqual(data["args"][:2], ["-m", module])
        self.assertIn(
            "data/with spaces",
            json.loads(self.run_script("generate_data.sh", "export", "--output", "data/with spaces").stdout)["args"],
        )

    def test_relative_overrides_and_explicit_python_precedence(self):
        custom = self.make_env("environments/export custom")
        for variables in (
            {"LEROBOT_VENV": "environments/export custom"},
            {"LEROBOT_VENV": "missing", "LEROBOT_PYTHON": "environments/export custom/bin/python"},
            {"LEROBOT_PYTHON": str(custom / "bin/python")},
        ):
            result = self.run_script("run-lerobot.sh", "validator.py", "--dataset", "data/set", **variables)
            data = json.loads(result.stdout)
            self.assertEqual(data["venv"], str(custom))
            self.assertEqual(data["args"], ["validator.py", "--dataset", "data/set"])

    def test_simulator_commands_select_the_simulator(self):
        custom = self.make_env("my simulator")
        for script, args in (("generate_data.sh", ["dev", "inspect"]), ("datagen-uv.sh", ["inventory"])):
            result = self.run_script(script, *args, ROBODOJO_VENV="my simulator")
            data = json.loads(result.stdout)
            self.assertEqual(data["venv"], str(custom))
            self.assertEqual(data["cwd"], str(self.root))
            self.assertIsNone(data["conda"])

    def test_missing_environment_is_actionable_and_never_falls_back(self):
        result = self.run_script("generate_data.sh", "export", expected=1, LEROBOT_VENV="missing")
        self.assertIn("install-lerobot-uv.sh", result.stderr)
        self.assertIn(str(self.root / "missing/bin/python"), result.stderr)

    def test_export_preserves_exit_code(self):
        self.run_script("generate_data.sh", "export", expected=27, TEST_STATUS="27")

    def test_installer_rejects_simulator_and_aliases_before_install(self):
        (self.root / "alias").symlink_to(self.root / ".venv", target_is_directory=True)
        for path in (".venv", "alias", "./sub/../.venv", ".venv/nested", "."):
            result = self.run_script("install-lerobot-uv.sh", "--venv", path, expected=1)
            self.assertIn("must be separate", result.stderr)

    def test_verify_missing_environment_does_not_install(self):
        result = self.run_script("install-lerobot-uv.sh", "--venv", "missing", "--verify-only", expected=1)
        self.assertIn("Run without --verify-only", result.stderr)
        self.assertFalse((self.root / "missing").exists())

    def test_installer_help_needs_no_python(self):
        shutil.rmtree(self.root / ".venv-lerobot")
        self.run_script("install-lerobot-uv.sh", "--help")


if __name__ == "__main__":
    unittest.main()
