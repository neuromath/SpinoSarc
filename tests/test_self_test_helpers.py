import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SELF_TEST_PATH = (
    Path(__file__).parents[1] / "spinosarc_app" / "self_test.py")
SPEC = importlib.util.spec_from_file_location(
    "spinosarc_self_test", SELF_TEST_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
_find_checkpoint = MODULE._find_checkpoint
_find_dataset = MODULE._find_dataset


class FrozenSelfTestHelperTests(unittest.TestCase):
    def test_release_nested_dataset_is_found(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "nnUNet" / "results"
            expected = (
                results / "r20260807" /
                "Dataset101_TotalSpineSeg_step1"
            )
            expected.mkdir(parents=True)
            self.assertEqual(
                _find_dataset(results, "Dataset101_TotalSpineSeg_step1"),
                expected,
            )

    def test_best_checkpoint_is_preferred(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "Dataset101_TotalSpineSeg_step1"
            fold = dataset / "trainer__plans__3d_fullres" / "fold_0"
            fold.mkdir(parents=True)
            final = fold / "checkpoint_final.pth"
            best = fold / "checkpoint_best.pth"
            final.touch()
            best.touch()
            self.assertEqual(_find_checkpoint(dataset), best)

    def test_missing_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                _find_checkpoint(Path(directory))


if __name__ == "__main__":
    unittest.main()
