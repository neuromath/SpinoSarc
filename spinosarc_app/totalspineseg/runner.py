"""TotalSpineSeg process runner for source and bundled macOS builds.

The release application contains TotalSpineSeg and its model weights.  A
second invocation of the SpinoSarc executable is used as an isolated worker,
which keeps nnU-Net state and memory out of the GUI process.  Development
installations can still use an in-environment TotalSpineSeg or the legacy
Conda environment.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional


class TotalSpineSegRunner:
    """Run TotalSpineSeg without requiring end users to install Conda."""

    WORKER_FLAG = "--spinosarc-tss-worker"

    def __init__(self, conda_env_name: str = "totalspineseg"):
        self.conda_env_name = conda_env_name
        self._backend: Optional[list[str]] = None

    def _find_conda(self) -> Optional[str]:
        conda = shutil.which("conda")
        if conda:
            return conda
        for candidate in (
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda",
            os.path.expanduser("~/anaconda3/bin/conda"),
            os.path.expanduser("~/miniconda3/bin/conda"),
        ):
            if os.path.isfile(candidate):
                return candidate
        return None

    def _resolve_backend(self) -> Optional[list[str]]:
        """Return the command prefix for the best available runtime."""
        if self._backend is not None:
            return list(self._backend)

        override = os.environ.get("SPINOSARC_TSS_EXECUTABLE")
        if override and Path(override).is_file():
            self._backend = [override]
            return list(self._backend)

        # PyInstaller release: invoke this app's executable in worker mode.
        if getattr(sys, "frozen", False):
            self._backend = [sys.executable, self.WORKER_FLAG]
            return list(self._backend)

        # Developer install with TotalSpineSeg in the active Python env.
        if importlib.util.find_spec("totalspineseg.inference") is not None:
            self._backend = [sys.executable, "-m", "totalspineseg.inference"]
            return list(self._backend)

        # Backward-compatible fallback for existing developer machines.
        conda = self._find_conda()
        if conda:
            self._backend = [
                conda, "run", "-n", self.conda_env_name, "totalspineseg"
            ]
            return list(self._backend)
        return None

    def is_available(self) -> bool:
        backend = self._resolve_backend()
        if not backend:
            return False
        try:
            result = subprocess.run(
                [*backend, "--help"],
                capture_output=True,
                text=True,
                timeout=60,
                env=self._worker_env(),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        return result.returncode == 0

    @staticmethod
    def _worker_env() -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        return env

    @staticmethod
    def _worker_limits() -> tuple[int, int]:
        """Conservative worker counts for 8–16 GB Apple Silicon Macs."""
        try:
            import psutil

            memory_gb = psutil.virtual_memory().total / 2**30
        except Exception:
            memory_gb = 8.0
        general = 1 if memory_gb < 12 else 2 if memory_gb < 24 else 4
        return general, 1

    def _command(
        self,
        sagittal_nifti_path: str,
        output_dir: str,
        device: str,
        step1_only: bool,
        iso: bool,
    ) -> list[str]:
        backend = self._resolve_backend()
        if not backend:
            return []
        max_workers, max_workers_nnunet = self._worker_limits()
        cmd = [
            *backend,
            str(Path(sagittal_nifti_path).resolve()),
            str(Path(output_dir).resolve()),
            "--device", device,
            "--max-workers", str(max_workers),
            "--max-workers-nnunet", str(max_workers_nnunet),
            "--quiet",
        ]
        if step1_only:
            cmd.append("--step1")
        if iso:
            cmd.append("--iso")
        return cmd

    @staticmethod
    def _mps_failure(stderr: str) -> bool:
        lowered = stderr.lower()
        markers = (
            "mps backend",
            "not implemented for 'mps'",
            "not implemented for mps",
            "placeholder storage has not been allocated on mps",
            "mps device",
        )
        return any(marker in lowered for marker in markers)

    def run(
        self,
        sagittal_nifti_path: str,
        output_dir: str,
        device: str = "mps",
        step1_only: bool = True,
        iso: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        def emit(message: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(message)
                except Exception:
                    pass

        sag = Path(sagittal_nifti_path)
        if not sag.is_file():
            return self._error(f"Sagittal NIfTI not found: {sag}")
        if sag.suffix != ".nii" and "".join(sag.suffixes[-2:]) != ".nii.gz":
            return self._error(f"Input is not a NIfTI file: {sag}")
        if not self._resolve_backend():
            return self._error("Bundled TotalSpineSeg runtime is unavailable")

        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        def execute(selected_device: str) -> tuple[subprocess.CompletedProcess, list[str], float]:
            cmd = self._command(str(sag), str(out), selected_device, step1_only, iso)
            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,
                env=self._worker_env(),
            )
            return result, cmd, time.time() - start

        emit("Starting lumbar level detection…")
        try:
            result, cmd, duration = execute(device)
            if result.returncode != 0 and device == "mps" and self._mps_failure(result.stderr):
                emit("Metal acceleration unavailable; retrying on CPU…")
                result, cmd, retry_duration = execute("cpu")
                duration += retry_duration
        except subprocess.TimeoutExpired as exc:
            return self._error(
                "Level detection exceeded the 15-minute safety limit.",
                stdout=(exc.stdout or "")[-4000:] if isinstance(exc.stdout, str) else "",
                stderr=(exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
                duration_sec=900.0,
            )
        except (FileNotFoundError, OSError) as exc:
            return self._error(f"Could not start level detection: {exc}")

        if result.returncode != 0:
            emit("Lumbar level detection failed")
            return self._error(
                f"TotalSpineSeg exited with code {result.returncode}.",
                stdout=(result.stdout or "")[-4000:],
                stderr=(result.stderr or "")[-4000:],
                command=cmd,
                duration_sec=duration,
            )

        expected = (out / "step1_levels", out / "step2_output")
        if not any(path.is_dir() for path in expected):
            return self._error(
                "Level detection finished but its output folder is missing.",
                stdout=(result.stdout or "")[-4000:],
                stderr=(result.stderr or "")[-4000:],
                command=cmd,
                duration_sec=duration,
            )

        emit(f"Lumbar level detection completed in {duration:.1f}s")
        return {
            "success": True,
            "output_dir": str(out),
            "duration_sec": duration,
            "command": cmd,
        }

    @staticmethod
    def _error(
        message: str,
        stdout: str = "",
        stderr: str = "",
        command: Optional[list[str]] = None,
        duration_sec: float = 0.0,
    ) -> dict:
        return {
            "success": False,
            "error": message,
            "stdout": stdout,
            "stderr": stderr,
            "command": command or [],
            "duration_sec": duration_sec,
        }
