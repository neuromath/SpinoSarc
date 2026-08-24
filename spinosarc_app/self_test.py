"""Release-time checks for the two bundled inference runtimes.

These checks are invoked against the frozen ``SpinoSarc.app`` after the
PyInstaller build.  They intentionally go beyond ``--help``: TotalSpineSeg's
actual checkpoint is deserialized and MuscleMap performs a real CPU forward
pass on a deterministic synthetic image.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path


log = logging.getLogger(__name__)


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def _totalspineseg_data_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "totalspineseg_data"
    configured = os.environ.get("SPINOSARC_TSS_DATA_BUILD")
    if configured:
        return Path(configured)
    return _bundle_root() / "totalspineseg_data"


def standalone_runtime_preflight() -> dict:
    """Check GUI imports, Excel support, and the bundled DICOM converter."""
    import openpyxl
    import PyQt6

    from . import gui
    from .dicom_loader import _resolve_dcm2niix

    converter = _resolve_dcm2niix()
    if not converter or not Path(converter).is_file():
        raise FileNotFoundError("Bundled dcm2niix executable is missing")
    completed = subprocess.run(
        [converter, "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Bundled dcm2niix did not start: "
            f"{completed.stderr or completed.stdout}"
        )
    version_output = (completed.stdout or completed.stderr or "").strip()
    if not hasattr(gui, "SpinoSarcWindow"):
        raise RuntimeError("SpinoSarc GUI entry point is missing")

    result = {
        "success": True,
        "dcm2niix": str(converter),
        "dcm2niix_version": version_output[-500:],
        "openpyxl_version": getattr(openpyxl, "__version__", "unknown"),
        "pyqt6_module": str(
            Path(PyQt6.__file__).resolve()
            if getattr(PyQt6, "__file__", None) else "namespace"
        ),
    }
    log.info("Standalone runtime preflight passed: %s", json.dumps(result))
    return result


def _find_dataset(results_root: Path, dataset_name: str) -> Path:
    candidates = sorted(
        path for path in results_root.rglob(dataset_name) if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"Bundled TotalSpineSeg dataset is missing: {dataset_name}"
        )
    return candidates[0]


def _find_checkpoint(dataset_dir: Path) -> Path:
    preferred = sorted(dataset_dir.rglob("fold_0/checkpoint_best.pth"))
    if not preferred:
        preferred = sorted(dataset_dir.rglob("fold_0/checkpoint_final.pth"))
    if not preferred:
        preferred = sorted(dataset_dir.rglob("fold_*/checkpoint*.pth"))
    if not preferred:
        raise FileNotFoundError(
            f"No readable nnU-Net checkpoint was found under {dataset_dir}"
        )
    return preferred[0]


def totalspineseg_preflight() -> dict:
    """Import the frozen runtime and deserialize its step-1 checkpoint."""
    import torch
    import totalspineseg
    import nnunetv2

    data_root = _totalspineseg_data_root()
    results_root = data_root / "nnUNet" / "results"
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"Bundled TotalSpineSeg results folder is missing: {results_root}"
        )

    datasets = {}
    for dataset_name in (
        "Dataset101_TotalSpineSeg_step1",
        "Dataset102_TotalSpineSeg_step2",
    ):
        dataset_dir = _find_dataset(results_root, dataset_name)
        checkpoint = _find_checkpoint(dataset_dir)
        if not list(dataset_dir.rglob("plans.json")):
            raise FileNotFoundError(f"plans.json is missing under {dataset_dir}")
        if not list(dataset_dir.rglob("dataset.json")):
            raise FileNotFoundError(f"dataset.json is missing under {dataset_dir}")
        datasets[dataset_name] = {
            "path": str(dataset_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
        }

    # TotalSpineSeg's step 1 is the path used by Detect Levels.  First
    # deserialize the checkpoint, then ask nnU-Net to reconstruct the actual
    # network/trainer from the model folder.  The latter exercises the dynamic
    # imports that simple ``--help`` smoke tests miss in frozen applications.
    step1_checkpoint = Path(
        datasets["Dataset101_TotalSpineSeg_step1"]["checkpoint"])
    try:
        checkpoint_data = torch.load(
            step1_checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint_data = torch.load(step1_checkpoint, map_location="cpu")
    if not isinstance(checkpoint_data, dict) or not checkpoint_data:
        raise ValueError(
            f"Unexpected checkpoint payload in {step1_checkpoint}"
        )
    checkpoint_keys = sorted(str(key) for key in checkpoint_data)[:12]
    del checkpoint_data
    gc.collect()

    from auglab.add_trainer import add_trainer
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    add_trainer("nnUNetTrainerDAExt")
    torch.set_num_threads(1)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    model_folder = step1_checkpoint.parent.parent
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0,),
        checkpoint_name=step1_checkpoint.name,
    )
    network_class = type(predictor.network).__name__
    del predictor
    gc.collect()

    result = {
        "success": True,
        "data_root": str(data_root),
        "datasets": datasets,
        "checkpoint_keys": checkpoint_keys,
        "network_class": network_class,
        "totalspineseg_module": str(
            Path(totalspineseg.__file__).resolve()
            if getattr(totalspineseg, "__file__", None) else "namespace"
        ),
        "nnunetv2_module": str(
            Path(nnunetv2.__file__).resolve()
            if getattr(nnunetv2, "__file__", None) else "namespace"
        ),
    }
    log.info("TotalSpineSeg preflight passed: %s", json.dumps(result))
    return result


def musclemap_inference_self_test() -> dict:
    """Perform a real bundled MuscleMap CPU inference on a small phantom."""
    import nibabel as nib
    import numpy as np

    from .inference_engine import MuscleMapEngine

    yy, xx = np.mgrid[-1.0:1.0:256j, -1.0:1.0:256j]
    image = (
        900.0 * np.exp(-((xx / 0.72) ** 2 + (yy / 0.88) ** 2) * 2.2)
        + 140.0 * np.exp(-(((xx - 0.32) / 0.18) ** 2
                           + ((yy + 0.08) / 0.30) ** 2) * 2.0)
        + 120.0 * np.exp(-(((xx + 0.32) / 0.18) ** 2
                           + ((yy + 0.08) / 0.30) ** 2) * 2.0)
    ).astype(np.float32)
    image[image < 1.0] = 0.0
    volume = image[:, :, None]

    with tempfile.TemporaryDirectory(prefix="spinosarc_self_test_") as directory:
        input_path = Path(directory) / "musclemap_phantom.nii.gz"
        affine = np.diag([1.2, 1.2, 5.0, 1.0])
        nib.save(nib.Nifti1Image(volume, affine), str(input_path))

        engine = MuscleMapEngine(use_gpu=False)
        segmentation = np.squeeze(engine.segment(str(input_path)))

    if segmentation.shape != image.shape:
        raise ValueError(
            "MuscleMap self-test shape mismatch: "
            f"segmentation={segmentation.shape}, input={image.shape}"
        )
    if not np.isfinite(segmentation).all():
        raise ValueError("MuscleMap self-test returned non-finite labels")
    labels = sorted(int(value) for value in np.unique(segmentation))
    if labels and (labels[0] < 0 or labels[-1] >= engine.out_channels):
        raise ValueError(f"MuscleMap returned unexpected labels: {labels}")

    result = {
        "success": True,
        "shape": list(segmentation.shape),
        "labels": labels,
        "device": str(engine.device),
        "model_path": str(engine.model_path),
    }
    log.info("MuscleMap inference self-test passed: %s", json.dumps(result))
    return result
