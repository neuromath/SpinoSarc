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
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional


class TotalSpineSegRunner:
    """Run TotalSpineSeg without requiring end users to install Conda."""

    WORKER_FLAG = "--spinosarc-tss-worker"
    PREFLIGHT_FLAG = "--spinosarc-tss-preflight"

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
        command = [*backend, "--help"]
        if getattr(sys, "frozen", False):
            command = [sys.executable, self.PREFLIGHT_FLAG]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=180,
                env=self._worker_env("cpu"),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False
        return result.returncode == 0

    @staticmethod
    def _worker_env(device: str = "cpu") -> dict[str, str]:
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        if device == "cpu":
            threads = str(max(1, min(os.cpu_count() or 1, 6)))
            env["OMP_NUM_THREADS"] = threads
            env["VECLIB_MAXIMUM_THREADS"] = threads
        else:
            env["OMP_NUM_THREADS"] = "1"
            env["VECLIB_MAXIMUM_THREADS"] = "1"
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
            "--keep-only", "step1_output", "step1_canal", "step1_levels",
            "--quiet",
        ]
        if step1_only:
            cmd.append("--step1")
        if iso:
            cmd.append("--iso")
        return cmd

    @staticmethod
    def _clear_output(path: Path) -> None:
        if path.exists():
            for child in path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink(missing_ok=True)
        path.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        sagittal_nifti_path: str,
        output_dir: str,
        device: str = "cpu",
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
        self._clear_output(out)

        def execute(selected_device: str):
            cmd = self._command(str(sag), str(out), selected_device, step1_only, iso)
            start = time.time()
            timeout_seconds = 1800 if selected_device == "cpu" else 600
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self._worker_env(selected_device),
                start_new_session=True,
            )
            timed_out = False
            while True:
                elapsed = time.time() - start
                remaining = timeout_seconds - elapsed
                if remaining <= 0:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (AttributeError, ProcessLookupError, PermissionError):
                        process.kill()
                    stdout, stderr = process.communicate()
                    break
                try:
                    stdout, stderr = process.communicate(timeout=min(15, remaining))
                    break
                except subprocess.TimeoutExpired:
                    minutes, seconds = divmod(int(elapsed), 60)
                    emit(
                        f"Detecting lumbar levels on {selected_device.upper()}… "
                        f"{minutes}m {seconds:02d}s"
                    )
            result = subprocess.CompletedProcess(
                cmd, process.returncode,
                stdout=stdout or "", stderr=stderr or "",
            )
            return result, cmd, time.time() - start, timed_out

        emit("Starting reliable CPU level detection…" if device == "cpu"
             else "Starting accelerated lumbar level detection…")
        attempts = []
        try:
            result, cmd, duration, timed_out = execute(device)
            attempts.append((device, result, duration, timed_out))
            # TotalSpineSeg officially supports CPU/CUDA. If an optional MPS
            # attempt fails for any reason, discard partial output and retry
            # deterministically on CPU.
            if device == "mps" and (timed_out or result.returncode != 0):
                emit("Metal attempt failed; retrying safely on CPU…")
                self._clear_output(out)
                result, cmd, retry_duration, timed_out = execute("cpu")
                attempts.append(("cpu", result, retry_duration, timed_out))
                duration += retry_duration
        except (FileNotFoundError, OSError) as exc:
            return self._error(f"Could not start level detection: {exc}")

        if timed_out:
            return self._error(
                "Level detection exceeded the 30-minute CPU safety limit.",
                stdout=(result.stdout or "")[-8000:],
                stderr=(result.stderr or "")[-8000:],
                command=cmd,
                duration_sec=duration,
            )

        if result.returncode != 0:
            emit("Lumbar level detection failed")
            previous = ""
            if len(attempts) > 1:
                first_device, first_result, _, _ = attempts[0]
                previous = (
                    f"\nPrevious {first_device} attempt:\n"
                    f"{(first_result.stderr or first_result.stdout or '')[-3000:]}"
                )
            return self._error(
                f"TotalSpineSeg exited with code {result.returncode}.",
                stdout=(result.stdout or "")[-8000:],
                stderr=((result.stderr or "")[-8000:] + previous),
                command=cmd,
                duration_sec=duration,
            )

        required_outputs = {
            "step1_levels": out / "step1_levels",
            "step1_canal": out / "step1_canal",
        }
        missing_outputs = [
            name for name, folder in required_outputs.items()
            if not folder.is_dir() or not list(folder.glob("*.nii*"))
        ]
        if missing_outputs:
            return self._error(
                "Level detection finished but required output is missing or "
                f"empty: {', '.join(missing_outputs)}.",
                stdout=(result.stdout or "")[-8000:],
                stderr=(result.stderr or "")[-8000:],
                command=cmd,
                duration_sec=duration,
            )

        emit(f"Lumbar level detection completed in {duration:.1f}s")
        return {
            "success": True,
            "output_dir": str(out),
            "duration_sec": duration,
            "command": cmd,
            "device": attempts[-1][0],
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
