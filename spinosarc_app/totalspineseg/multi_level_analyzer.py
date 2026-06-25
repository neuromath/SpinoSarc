"""
MultiLevelAnalyzer: run paraspinal muscle + dural sac analysis across all
covered lumbar IVD levels, plus L3-based sarcopenia.

Single responsibility: orchestrate per-level analysis by reusing the existing
single-slice `analyzer.analyze()` and the canal CSA / stenosis helpers.

For each covered IVD level (L1-L2 ... L5-S):
  - paraspinal muscle metrics (CSA, fat fraction) via MuscleMap analyzer
  - dural sac CSA via the precomputed TotalSpineSeg canal NIfTI
  - stenosis flag (neutral, literature thresholds)

For L3 (vertebral body level, "L3_body"):
  - full sarcopenia (PMI, TPA, risk) -- this is the ONLY level where PMI is
    computed, per the L3 standard in sarcopenia literature.
  - if L3_body has no axial slice (gap), the nearest covered slice is used.

Does NOT:
  - Run TotalSpineSeg (runner.py's job; canal NIfTI must already exist)
  - Build reports (multi_level_report.py's job)
  - Touch the GUI

Design note: muscle analysis reuses the GUI's proven per-slice dcm2niix path
(one source DICOM -> NIfTI -> analyze). The caller supplies a callable that
produces a single-slice NIfTI for a given axial slice index, so this module
stays decoupled from how slices are loaded.
"""

from __future__ import annotations

from typing import Callable, Optional

from .canal_csa import (
    resample_canal_to_axial_slice,
    classify_stenosis,
    DEFAULT_THRESHOLD,
)


# IVD levels we analyze for muscle + canal (sarcopenia handled separately at L3)
IVD_LEVELS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S"]
SARCOPENIA_LEVEL = "L3_body"


class MultiLevelAnalyzer:
    """Run muscle + canal + stenosis analysis across covered lumbar levels."""

    def __init__(self, analyzer, axial_slices, canal_nifti_path):
        """
        Parameters
        ----------
        analyzer
            The existing SpinoSarc analyzer (has .analyze(slice_path, demo)).
        axial_slices
            SpinoSarc's axial_slices list (each has image_position,
            image_orientation, pixel_spacing, pixel_array, source_path).
        canal_nifti_path
            Path to step1_canal/*.nii.gz from a completed Detect Levels run.
        """
        self.analyzer = analyzer
        self.axial_slices = axial_slices
        self.canal_nifti_path = canal_nifti_path

    # ------------------------------------------------------------------
    def analyze_all(
        self,
        detected_levels: dict,
        slice_nifti_producer: Callable[[int], Optional[str]],
        demographics=None,
        canal_threshold: float = DEFAULT_THRESHOLD,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """Analyze every covered level.

        Parameters
        ----------
        detected_levels
            Output of LevelMapper.parse() + map_to_axial(). Each value has
            world_xyz, axial_slice_idx (None if out of range), type.
        slice_nifti_producer
            Callable: axial_slice_idx -> path to a single-slice NIfTI (or None
            on failure). The caller implements this using its own dcm2niix flow.
        demographics
            Optional Demographics object for sarcopenia (height needed for PMI).
        canal_threshold
            Soft mask threshold for canal CSA.
        progress_callback
            Optional callable(str) for progress messages.

        Returns
        -------
        dict
            {
              "levels": {
                 "L1-L2": {
                    "axial_slice_idx": int | None,
                    "covered": bool,
                    "muscles": [ {name, csa_mm2, csa_cm2, fat_fraction, ...}, ... ],
                    "asymmetry": {...},
                    "canal_csa_mm2": float | None,
                    "stenosis": {category, label, flag, caveat} | None,
                    "error": str (only if something failed),
                 },
                 ...
              },
              "sarcopenia": {
                 "level_used": "L3_body" | "<nearest>",
                 "axial_slice_idx": int,
                 "result": {... SarcopeniaResult asdict ...},
                 "note": str,
              } | None,
            }
        """
        def _progress(msg):
            if progress_callback:
                progress_callback(msg)

        out = {"levels": {}, "sarcopenia": None}

        # ---- Per-IVD muscle + canal + stenosis ----
        for level_name in IVD_LEVELS:
            info = detected_levels.get(level_name)
            if info is None:
                # Level not detected at all (shouldn't usually happen for IVDs)
                out["levels"][level_name] = {
                    "covered": False,
                    "axial_slice_idx": None,
                    "error": "level not detected",
                }
                continue

            ax_idx = info.get("axial_slice_idx")
            entry = {
                "axial_slice_idx": ax_idx,
                "covered": ax_idx is not None,
            }

            # --- Canal CSA (works even if muscle slice is missing) ---
            canal_csa = self._canal_csa_for_slice(ax_idx, canal_threshold)
            entry["canal_csa_mm2"] = canal_csa
            if canal_csa is not None:
                entry["stenosis"] = classify_stenosis(canal_csa, level_name)
            else:
                entry["stenosis"] = None

            # --- Muscle metrics (needs an axial slice) ---
            # If this IVD is in a gap, use the nearest slice and flag it as
            # approximate, recomputing canal CSA at that nearest slice too.
            approx = False
            approx_dist = None
            if ax_idx is None:
                world_z = info.get("world_xyz", (0, 0, 0))[2]
                near_idx, near_dist = self._nearest_slice_idx(world_z)
                if near_idx is None:
                    entry["muscles"] = []
                    entry["asymmetry"] = {}
                    entry["note"] = "no axial slices available"
                    out["levels"][level_name] = entry
                    continue
                ax_idx = near_idx
                approx = True
                approx_dist = near_dist
                entry["approx"] = True
                entry["approx_distance_mm"] = near_dist
                entry["axial_slice_idx"] = ax_idx
                # Recompute canal CSA + stenosis at the nearest slice
                canal_csa = self._canal_csa_for_slice(ax_idx, canal_threshold)
                entry["canal_csa_mm2"] = canal_csa
                if canal_csa is not None:
                    entry["stenosis"] = classify_stenosis(canal_csa, level_name)

            label_suffix = (f" (approx: nearest slice {approx_dist:.1f} mm away)"
                            if approx else "")
            _progress(f"Analyzing muscles at {level_name} "
                      f"(slice {ax_idx + 1}){label_suffix}...")
            try:
                slice_path = slice_nifti_producer(ax_idx)
                if not slice_path:
                    entry["error"] = "could not produce slice NIfTI"
                    entry["muscles"] = []
                    out["levels"][level_name] = entry
                    continue
                result = self.analyzer.analyze(slice_path, demographics)
                entry["muscles"] = result.get("muscles", [])
                entry["asymmetry"] = result.get("asymmetry", {})
            except Exception as e:
                import traceback
                traceback.print_exc()
                entry["error"] = f"muscle analysis failed: {e}"
                entry["muscles"] = []

            out["levels"][level_name] = entry

        # ---- L3 sarcopenia ----
        out["sarcopenia"] = self._analyze_sarcopenia(
            detected_levels, slice_nifti_producer, demographics, _progress
        )

        return out

    # ------------------------------------------------------------------
    def _nearest_slice_idx(self, world_z):
        """Find the axial slice index whose z is closest to world_z.
        Returns (idx, distance_mm) or (None, None)."""
        best_idx, best_dist = None, float("inf")
        for i, s in enumerate(self.axial_slices):
            ipp = s.get("image_position")
            if ipp is None or len(ipp) < 3:
                continue
            d = abs(float(ipp[2]) - float(world_z))
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is None:
            return None, None
        return best_idx, best_dist

    # ------------------------------------------------------------------
    def _canal_csa_for_slice(self, ax_idx, threshold):
        """Canal CSA (mm^2) at an axial slice index, or None."""
        if ax_idx is None:
            return None
        if not self.canal_nifti_path:
            return None
        if ax_idx < 0 or ax_idx >= len(self.axial_slices):
            return None
        try:
            mask = resample_canal_to_axial_slice(
                self.canal_nifti_path,
                self.axial_slices[ax_idx],
                threshold=threshold,
            )
            if mask is None:
                return None
            voxel_count = int(mask.sum())
            ps = self.axial_slices[ax_idx].get("pixel_spacing", (1.0, 1.0))
            pixel_area = float(ps[0]) * float(ps[1])
            return float(voxel_count * pixel_area)
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    def _analyze_sarcopenia(
        self, detected_levels, slice_nifti_producer, demographics, progress
    ):
        """Compute sarcopenia at L3_body (or nearest covered slice)."""
        l3 = detected_levels.get(SARCOPENIA_LEVEL)
        if l3 is None:
            return None

        ax_idx = l3.get("axial_slice_idx")
        note = "L3 vertebral body level"

        # If L3_body has no axial slice (gap), find the nearest covered slice
        # by world-z to the L3_body z.
        if ax_idx is None:
            l3_z = l3.get("world_xyz", (0, 0, 0))[2]
            best_idx = None
            best_dist = float("inf")
            for i, s in enumerate(self.axial_slices):
                ipp = s.get("image_position")
                if ipp is None or len(ipp) < 3:
                    continue
                d = abs(float(ipp[2]) - float(l3_z))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is None:
                return {
                    "level_used": SARCOPENIA_LEVEL,
                    "axial_slice_idx": None,
                    "result": None,
                    "note": "L3 in gap and no axial slices available",
                }
            ax_idx = best_idx
            note = (f"L3_body in axial gap; used nearest slice "
                    f"{ax_idx + 1} ({best_dist:.1f} mm away)")

        progress(f"Analyzing sarcopenia at L3 (slice {ax_idx + 1})...")
        try:
            slice_path = slice_nifti_producer(ax_idx)
            if not slice_path:
                return {
                    "level_used": SARCOPENIA_LEVEL,
                    "axial_slice_idx": ax_idx,
                    "result": None,
                    "note": note + " - could not produce slice NIfTI",
                }
            result = self.analyzer.analyze(slice_path, demographics)
            return {
                "level_used": SARCOPENIA_LEVEL,
                "axial_slice_idx": ax_idx,
                "result": result.get("sarcopenia"),
                "muscles": result.get("muscles", []),
                "note": note,
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "level_used": SARCOPENIA_LEVEL,
                "axial_slice_idx": ax_idx,
                "result": None,
                "note": note + f" - sarcopenia failed: {e}",
            }
