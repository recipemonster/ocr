from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bootstrap


class BootstrapTest(unittest.TestCase):
    def test_install_runtime_uses_uv_and_publishes_complete_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app_directory = root / "app"
            app_directory.mkdir()
            (app_directory / "requirements-base.txt").write_text("Flask==3.1.3\n", encoding="utf-8")
            (app_directory / "requirements-cpu.txt").write_text("paddlepaddle==3.3.0\n", encoding="utf-8")
            destination = root / "runtime" / "cpu-test"
            destination.parent.mkdir()
            commands = []

            def run(command, **_kwargs):
                commands.append(command)
                if command[:2] == ["uv", "venv"]:
                    (Path(command[-1]) / "bin").mkdir(parents=True)
                return subprocess.CompletedProcess(command, 0)

            with (
                patch.object(bootstrap, "APP_DIRECTORY", app_directory),
                patch.object(bootstrap, "PACKAGE_CACHE_DIRECTORY", root / "uv-cache"),
                patch.object(bootstrap.subprocess, "run", side_effect=run),
            ):
                bootstrap.install_runtime("cpu", destination)

            self.assertEqual(commands[0][:2], ["uv", "venv"])
            self.assertIn("--no-managed-python", commands[0])
            self.assertEqual(commands[1][:3], ["uv", "pip", "install"])
            self.assertIn("--index-strategy", commands[1])
            self.assertNotIn("pip-cache", " ".join(commands[1]))
            self.assertEqual(commands[2][1], str(app_directory / "license_inventory.py"))
            self.assertEqual(commands[2][-2], "--output")
            self.assertEqual(Path(commands[2][-1]).name, "licenses")
            self.assertTrue((destination / "runtime.json").is_file())

    def test_failed_install_does_not_publish_partial_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app_directory = root / "app"
            app_directory.mkdir()
            (app_directory / "requirements-base.txt").write_text("Flask==3.1.3\n", encoding="utf-8")
            (app_directory / "requirements-cpu.txt").write_text("paddlepaddle==3.3.0\n", encoding="utf-8")
            destination = root / "runtime" / "cpu-test"
            destination.parent.mkdir()
            command_count = 0

            def run(command, **_kwargs):
                nonlocal command_count
                command_count += 1
                if command_count == 1:
                    (Path(command[-1]) / "bin").mkdir(parents=True)
                    return subprocess.CompletedProcess(command, 0)
                raise subprocess.CalledProcessError(1, command)

            with (
                patch.object(bootstrap, "APP_DIRECTORY", app_directory),
                patch.object(bootstrap, "PACKAGE_CACHE_DIRECTORY", root / "uv-cache"),
                patch.object(bootstrap.subprocess, "run", side_effect=run),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    bootstrap.install_runtime("cpu", destination)

            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
