"""SpinoSarc application and bundled TotalSpineSeg worker entry point."""

from __future__ import annotations

import os
import sys
import multiprocessing
import json
import logging
import traceback
from pathlib import Path


WORKER_FLAG = "--spinosarc-tss-worker"
RUNTIME_PREFLIGHT_FLAG = "--spinosarc-runtime-preflight"
TSS_PREFLIGHT_FLAG = "--spinosarc-tss-preflight"
MUSCLEMAP_SELF_TEST_FLAG = "--spinosarc-musclemap-self-test"


def _configure_diagnostics() -> Path:
    """Persist diagnostics for Finder-launched windowed applications."""
    log_dir = Path.home() / "Library" / "Logs" / "SpinoSarc"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "SpinoSarc.log"
    os.environ["SPINOSARC_LOG_PATH"] = str(log_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )

    # PyInstaller's windowed bootloader normally sets these to None.  Private
    # worker/self-test modes are launched with real file descriptors, so keep
    # their output capturable by the parent process.  The Finder GUI falls back
    # to the persistent log file.
    private_mode = any(flag in sys.argv for flag in (
        WORKER_FLAG, RUNTIME_PREFLIGHT_FLAG,
        TSS_PREFLIGHT_FLAG, MUSCLEMAP_SELF_TEST_FLAG))
    for name, descriptor in (("stdout", 1), ("stderr", 2)):
        if getattr(sys, name) is not None:
            continue
        stream = None
        if private_mode:
            try:
                stream = os.fdopen(
                    os.dup(descriptor), "w", buffering=1, encoding="utf-8")
            except OSError:
                stream = None
        if stream is None:
            stream = open(log_path, "a", buffering=1, encoding="utf-8")
        setattr(sys, name, stream)

    def _log_unhandled(exc_type, exc_value, exc_traceback):
        logging.getLogger("spinosarc.unhandled").critical(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        traceback.print_exception(exc_type, exc_value, exc_traceback)

    sys.excepthook = _log_unhandled
    logging.getLogger(__name__).info(
        "SpinoSarc starting; argv=%s; frozen=%s",
        sys.argv, getattr(sys, "frozen", False),
    )
    return log_path


def _run_self_test(name, function) -> int:
    try:
        result = function()
        print(json.dumps({"self_test": name, **result}, sort_keys=True))
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("%s failed", name)
        print(f"{name} failed: {exc}", file=sys.stderr)
        return 1


def _run_totalspineseg_worker() -> None:
    # Remove the private dispatcher argument before TotalSpineSeg parses argv.
    sys.argv.remove(WORKER_FLAG)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # TotalSpineSeg otherwise expands CPU inference to every logical core,
    # which can make unified-memory Macs unresponsive.  The GUI runner sets a
    # conservative OMP limit; mirror it for TotalSpineSeg's explicit
    # ``multiprocessing.cpu_count()`` call as well.
    try:
        selected_device = sys.argv[sys.argv.index("--device") + 1]
    except (ValueError, IndexError):
        selected_device = "cpu"
    if selected_device == "cpu":
        real_cpu_count = multiprocessing.cpu_count
        thread_limit = max(1, min(
            int(os.environ.get("OMP_NUM_THREADS", "6")), real_cpu_count()))
        multiprocessing.cpu_count = lambda: thread_limit
        try:
            import torch

            torch.set_num_threads(thread_limit)
            torch.set_num_interop_threads(min(2, thread_limit))
        except Exception:
            logging.getLogger(__name__).exception(
                "Could not apply TotalSpineSeg CPU thread limits")

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        bundled_data = Path(sys._MEIPASS) / "totalspineseg_data"
        if bundled_data.is_dir() and "--data-dir" not in sys.argv and "-d" not in sys.argv:
            sys.argv.extend(["--data-dir", str(bundled_data)])

        # TotalSpineSeg normally copies nnUNetTrainerDAExt into nnunetv2 at
        # first inference.  The release embeds that trainer before signing;
        # replace the installer with an import check so the .app is immutable.
        import importlib

        trainer_module_name = (
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainerDAExt")
        importlib.import_module(trainer_module_name)
        trainer_installer = importlib.import_module("auglab.add_trainer")

        def _use_bundled_trainer(trainer_name, overwrite=False):
            if trainer_name != "nnUNetTrainerDAExt":
                raise ValueError(f"Unexpected nnU-Net trainer: {trainer_name}")
            return importlib.import_module(trainer_module_name)

        trainer_installer.add_trainer = _use_bundled_trainer

    from totalspineseg.inference import main as totalspineseg_main

    totalspineseg_main()


if __name__ == "__main__":
    _configure_diagnostics()
    multiprocessing.freeze_support()
    if WORKER_FLAG in sys.argv:
        _run_totalspineseg_worker()
    elif RUNTIME_PREFLIGHT_FLAG in sys.argv:
        from spinosarc_app.self_test import standalone_runtime_preflight

        raise SystemExit(_run_self_test(
            "Standalone runtime preflight", standalone_runtime_preflight))
    elif TSS_PREFLIGHT_FLAG in sys.argv:
        from spinosarc_app.self_test import totalspineseg_preflight

        raise SystemExit(_run_self_test(
            "TotalSpineSeg preflight", totalspineseg_preflight))
    elif MUSCLEMAP_SELF_TEST_FLAG in sys.argv:
        from spinosarc_app.self_test import musclemap_inference_self_test

        raise SystemExit(_run_self_test(
            "MuscleMap inference", musclemap_inference_self_test))
    else:
        from spinosarc_app.gui import main

        main()
