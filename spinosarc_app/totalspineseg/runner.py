"""
TotalSpineSeg subprocess runner.

Calls TotalSpineSeg from a separate conda environment to avoid
dependency conflicts with SpinoSarc's own environment (MuscleMap
uses a different nnU-Net version).

Single responsibility: launch subprocess, manage errors, report progress.
NIfTI parsing and axial slice mapping happens elsewhere (level_mapper.py).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional


class TotalSpineSegRunner:
    """Wraps the `totalspineseg` CLI from a separate conda environment."""

    def __init__(self, conda_env_name: str = "totalspineseg"):
        self.conda_env_name = conda_env_name
        self._conda_exe: Optional[str] = None

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def _find_conda(self) -> Optional[str]:
        """Locate the conda executable.

        Returns absolute path to conda, or None if not found.
        Caches the result.
        """
        if self._conda_exe is not None:
            return self._conda_exe

        # 1) PATH lookup
        conda = shutil.which("conda")
        if conda:
            self._conda_exe = conda
            return conda

        # 2) Common install paths on macOS
        for candidate in (
            "/opt/anaconda3/bin/conda",
            "/opt/miniconda3/bin/conda",
            os.path.expanduser("~/anaconda3/bin/conda"),
            os.path.expanduser("~/miniconda3/bin/conda"),
        ):
            if os.path.isfile(candidate):
                self._conda_exe = candidate
                return candidate

        return None

    def is_available(self) -> bool:
        """Check whether `totalspineseg` is callable in the configured env.

        Returns True only if both conda is found and the env contains the
        `totalspineseg` executable. Quick to call - meant for startup
        checks (e.g. enable/disable a GUI button).
        """
        conda = self._find_conda()
        if conda is None:
            return False

        try:
            r = subprocess.run(
                [conda, "run", "-n", self.conda_env_name,
                 "totalspineseg", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

        return r.returncode == 0

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        sagittal_nifti_path: str,
        output_dir: str,
        device: str = "mps",
        step1_only: bool = True,
        iso: bool = True,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Run TotalSpineSeg on a sagittal NIfTI volume.

        Parameters
        ----------
        sagittal_nifti_path
            Absolute path to a sagittal NIfTI (`.nii` or `.nii.gz`).
            Must be a single file; TotalSpineSeg's input folder will be
            created internally.
        output_dir
            Absolute path where TotalSpineSeg will write its outputs.
            Will be created if missing. Existing contents are NOT erased.
        device
            "mps" (Apple Silicon), "cuda" (NVIDIA), or "cpu".
            Defaults to "mps". The TotalSpineSeg installation must have
            our MPS support patch applied for "mps" to work.
        step1_only
            If True, pass `--step1` (faster, vertebra-level identification only).
            If False, runs the full two-step pipeline (slower, individual labels).
        iso
            If True, pass `--iso` (1mm isotropic output, easier downstream
            processing). Defaults to True.
        progress_callback
            Optional callable taking a single status string. Called at
            start, intermediate milestones (if available), and end.

        Returns
        -------
        dict
            On success:
                {
                    "success": True,
                    "output_dir": <absolute path>,
                    "duration_sec": <float>,
                    "command": <list of strings>,
                }
            On failure:
                {
                    "success": False,
                    "error": <human-readable string>,
                    "stdout": <last 2000 chars>,
                    "stderr": <last 2000 chars>,
                    "command": <list of strings>,
                    "duration_sec": <float>,
                }
        """
        def _emit(msg: str) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    # progress callback errors should not break inference
                    pass

        # ------ validate input ------
        sag = Path(sagittal_nifti_path)
        if not sag.is_file():
            return {
                "success": False,
                "error": f"Sagittal NIfTI not found: {sagittal_nifti_path}",
                "stdout": "",
                "stderr": "",
                "command": [],
                "duration_sec": 0.0,
            }
        if sag.suffix not in (".nii",) and "".join(sag.suffixes[-2:]) != ".nii.gz":
            return {
                "success": False,
                "error": f"Input is not a NIfTI file: {sagittal_nifti_path}",
                "stdout": "",
                "stderr": "",
                "command": [],
                "duration_sec": 0.0,
            }

        # ------ resolve conda ------
        conda = self._find_conda()
        if conda is None:
            return {
                "success": False,
                "error": "conda executable not found",
                "stdout": "",
                "stderr": "",
                "command": [],
                "duration_sec": 0.0,
            }

        # ------ prepare a single-file input folder ------
        # TotalSpineSeg accepts either a single file OR a folder. To keep
        # the output deterministic (named by input filename), we always
        # pass the file directly. But we still need an output folder.
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)

        # ------ build command ------
        cmd = [
            conda, "run", "-n", self.conda_env_name,
            "totalspineseg",
            str(sag.resolve()),
            str(out),
            "--device", device,
        ]
        if step1_only:
            cmd.append("--step1")
        if iso:
            cmd.append("--iso")

        _emit("Starting TotalSpineSeg...")

        # ------ run ------
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min hard cap
            )
        except subprocess.TimeoutExpired as e:
            duration = time.time() - t0
            _emit("TotalSpineSeg timed out")
            return {
                "success": False,
                "error": (
                    f"TotalSpineSeg timed out after {duration:.0f} seconds. "
                    "On a Mac with MPS this should complete in ~1 minute. "
                    "On CPU it may exceed the 10-minute cap."
                ),
                "stdout": (e.stdout or "")[-2000:] if hasattr(e, "stdout") else "",
                "stderr": (e.stderr or "")[-2000:] if hasattr(e, "stderr") else "",
                "command": cmd,
                "duration_sec": duration,
            }
        except (FileNotFoundError, OSError) as e:
            duration = time.time() - t0
            return {
                "success": False,
                "error": f"Could not execute subprocess: {e}",
                "stdout": "",
                "stderr": "",
                "command": cmd,
                "duration_sec": duration,
            }

        duration = time.time() - t0

        if r.returncode != 0:
            _emit("TotalSpineSeg failed")
            return {
                "success": False,
                "error": (
                    f"TotalSpineSeg exited with code {r.returncode}. "
                    "See stdout/stderr below for details."
                ),
                "stdout": r.stdout[-2000:] if r.stdout else "",
                "stderr": r.stderr[-2000:] if r.stderr else "",
                "command": cmd,
                "duration_sec": duration,
            }

        # ------ verify expected output exists ------
        # step1_levels is the key output for level identification.
        # If --step1 was used the final levels folder may be named differently;
        # we look for both names.
        levels_dir_candidates = [
            out / "step1_levels",
            out / "step2_output",  # if step1_only=False
        ]
        if not any(p.is_dir() for p in levels_dir_candidates):
            _emit("TotalSpineSeg finished but expected output is missing")
            return {
                "success": False,
                "error": (
                    "TotalSpineSeg exited successfully but no expected output "
                    "directory (step1_levels or step2_output) was found in "
                    f"{out}"
                ),
                "stdout": r.stdout[-2000:] if r.stdout else "",
                "stderr": r.stderr[-2000:] if r.stderr else "",
                "command": cmd,
                "duration_sec": duration,
            }

        _emit(f"TotalSpineSeg completed in {duration:.1f}s")
        return {
            "success": True,
            "output_dir": str(out),
            "duration_sec": duration,
            "command": cmd,
        }
