import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


RUNNER_PATH = (
    Path(__file__).parents[1]
    / "spinosarc_app"
    / "totalspineseg"
    / "runner.py"
)
SPEC = importlib.util.spec_from_file_location("spinosarc_tss_runner", RUNNER_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
TotalSpineSegRunner = MODULE.TotalSpineSegRunner


class StandaloneRuntimeTests(unittest.TestCase):
    def test_frozen_runtime_uses_application_worker(self):
        runner = TotalSpineSegRunner()
        with mock.patch.object(sys, "frozen", True, create=True):
            self.assertEqual(
                runner._resolve_backend(),
                [sys.executable, "--spinosarc-tss-worker"],
            )

    def test_command_has_memory_safe_worker_limits(self):
        runner = TotalSpineSegRunner()
        runner._backend = ["SpinoSarc", "--spinosarc-tss-worker"]
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "sagittal.nii.gz"
            image.touch()
            command = runner._command(str(image), directory, "mps", True, True)
        self.assertIn("--max-workers-nnunet", command)
        self.assertIn("--step1", command)
        self.assertIn("--iso", command)
        self.assertIn("--keep-only", command)
        self.assertIn("step1_canal", command)
        self.assertIn("step1_levels", command)

    def test_cpu_is_the_default_totalspineseg_device(self):
        parameter = inspect.signature(TotalSpineSegRunner.run).parameters[
            "device"]
        self.assertEqual(parameter.default, "cpu")

    def test_clear_output_removes_partial_worker_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            partial = output / "step1_levels"
            partial.mkdir(parents=True)
            (partial / "partial.nii.gz").touch()
            (output / "stderr.txt").write_text("failed")

            TotalSpineSegRunner._clear_output(output)

            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
