"""
LevelMapper: parse TotalSpineSeg's `step1_levels` output and map each
disc level to (a) world coordinates and (b) the nearest axial DICOM slice.

Single responsibility: turn TotalSpineSeg's raw NIfTI label map into a
clean Python dict keyed by anatomical level name, and map each level to
the nearest axial slice index in SpinoSarc's existing slice list.

Does NOT:
- Run TotalSpineSeg (runner.py's job)
- Run MuscleMap inference (analyzer.py's job)
- Touch the GUI

Assumes runner.py has already produced a step1_levels NIfTI in the
TotalSpineSeg output directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib


# ---------------------------------------------------------------------------
# Constants from TotalSpineSeg's levels_maps.json
# (resources/labels_maps/levels_maps.json)
# These are TotalSpineSeg's compact level codes used in step1_levels NIfTI.
# ---------------------------------------------------------------------------

LEVELS_MAP = {
    1: "C1",
    2: "C1-C2",
    3: "C2-C3",
    4: "C3-C4",
    5: "C4-C5",
    6: "C5-C6",
    7: "C6-C7",
    8: "C7-T1",
    9: "T1-T2",
    10: "T2-T3",
    11: "T3-T4",
    12: "T4-T5",
    13: "T5-T6",
    14: "T6-T7",
    15: "T7-T8",
    16: "T8-T9",
    17: "T9-T10",
    18: "T10-T11",
    19: "T11-T12",
    20: "T12-L1",
    21: "L1-L2",
    22: "L2-L3",
    23: "L3-L4",
    24: "L4-L5",
    25: "L5-S",
}

# SpinoSarc-specific 6 target levels we want to analyze.
# 5 IVDs for stenosis assessment + L3 vertebral body for sarcopenia.
TARGET_IVDS = ["L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S"]
TARGET_LEVELS = TARGET_IVDS + ["L3_body"]


class LevelMapper:
    """Parse TotalSpineSeg level output and resolve axial slice indices."""

    LEVELS_MAP = LEVELS_MAP
    TARGET_IVDS = TARGET_IVDS
    TARGET_LEVELS = TARGET_LEVELS

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse(
        self,
        totalspineseg_output_dir: str,
        sagittal_nifti_path: str,
    ) -> dict:
        """Read the step1_levels NIfTI, return a level -> info dict.

        Parameters
        ----------
        totalspineseg_output_dir
            Directory produced by `runner.run()`. We expect a
            subdirectory `step1_levels/` containing exactly one NIfTI.
        sagittal_nifti_path
            Path to the original sagittal NIfTI that TotalSpineSeg was run
            on. Used to recover the corresponding levels file name (output
            files in step1_levels carry the same basename).

        Returns
        -------
        dict
            Keys are level names from TARGET_LEVELS that were actually
            found (IVDs not visible in the sagittal volume will be
            absent). Each value:
                {
                    "world_xyz": (x, y, z),       # mm in DICOM patient coords
                    "voxel_xyz": (i, j, k) | None,  # voxel index in the sagittal NIfTI
                    "type": "IVD" | "VB",
                    "source_label": <int> | "computed",
                }

            L3_body is added only if BOTH L2-L3 and L3-L4 were found
            (arithmetic midpoint).

        Raises
        ------
        FileNotFoundError
            If no step1_levels NIfTI is found.
        ValueError
            If the NIfTI is empty (no non-zero labels at all).
        """
        out = Path(totalspineseg_output_dir)
        levels_dir = out / "step1_levels"
        if not levels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected step1_levels directory under "
                f"{totalspineseg_output_dir}, but it does not exist."
            )

        # TotalSpineSeg names the output after the input basename.
        # E.g. input '2_t2_tse_sag_384_L.nii' -> '2_t2_tse_sag_384_L.nii.gz'.
        sag_path = Path(sagittal_nifti_path)
        sag_base = sag_path.name
        # Strip .nii / .nii.gz
        if sag_base.endswith(".nii.gz"):
            sag_stem = sag_base[: -len(".nii.gz")]
        elif sag_base.endswith(".nii"):
            sag_stem = sag_base[: -len(".nii")]
        else:
            sag_stem = sag_path.stem

        # Try both .nii.gz and .nii
        candidates = [
            levels_dir / f"{sag_stem}.nii.gz",
            levels_dir / f"{sag_stem}.nii",
        ]
        # Fallback: any single nifti in the folder
        levels_file: Optional[Path] = None
        for c in candidates:
            if c.is_file():
                levels_file = c
                break
        if levels_file is None:
            any_nii = list(levels_dir.glob("*.nii*"))
            if len(any_nii) == 1:
                levels_file = any_nii[0]
            else:
                raise FileNotFoundError(
                    f"Could not find levels NIfTI in {levels_dir}. "
                    f"Looked for {[c.name for c in candidates]}, "
                    f"found {[p.name for p in any_nii]}."
                )

        # Load
        img = nib.load(str(levels_file))
        data = np.asarray(img.get_fdata())
        affine = np.asarray(img.affine)

        unique_labels = [int(v) for v in np.unique(data) if int(v) != 0]
        if not unique_labels:
            raise ValueError(
                f"Levels NIfTI {levels_file} contains no non-zero labels."
            )

        # Build a level_name -> info dict for whatever labels are present.
        found: dict = {}
        for lbl in unique_labels:
            name = LEVELS_MAP.get(lbl)
            if name is None:
                # Unknown label code; skip silently rather than crash.
                continue
            mask = (data == lbl)
            if not mask.any():
                continue
            coords = np.argwhere(mask)
            # step1_levels stores single-voxel markers, but be defensive
            # and use the centroid in case of more than one voxel.
            vox = coords.mean(axis=0)  # (i, j, k) in float
            world_hom = affine @ np.array([vox[0], vox[1], vox[2], 1.0])
            world = world_hom[:3]
            found[name] = {
                "world_xyz": (float(world[0]), float(world[1]), float(world[2])),
                "voxel_xyz": (float(vox[0]), float(vox[1]), float(vox[2])),
                "type": "IVD",
                "source_label": lbl,
            }

        # Keep only the IVDs we actually care about.
        result: dict = {name: found[name] for name in TARGET_IVDS if name in found}

        # ------- L3 body (arithmetic midpoint of L2-L3 and L3-L4) -------
        if "L2-L3" in result and "L3-L4" in result:
            wx = (result["L2-L3"]["world_xyz"][0] + result["L3-L4"]["world_xyz"][0]) / 2.0
            wy = (result["L2-L3"]["world_xyz"][1] + result["L3-L4"]["world_xyz"][1]) / 2.0
            wz = (result["L2-L3"]["world_xyz"][2] + result["L3-L4"]["world_xyz"][2]) / 2.0
            result["L3_body"] = {
                "world_xyz": (wx, wy, wz),
                "voxel_xyz": None,
                "type": "VB",
                "source_label": "computed",
            }

        return result

    # ------------------------------------------------------------------
    # Axial mapping
    # ------------------------------------------------------------------

    def map_to_axial(
        self,
        levels: dict,
        axial_slice_metadata: list,
    ) -> dict:
        """Add `axial_slice_idx` to each level by finding the nearest axial slice.

        Parameters
        ----------
        levels
            Output of `parse()`. Each value's `world_xyz` is in DICOM
            patient coordinates (mm).
        axial_slice_metadata
            SpinoSarc's existing axial slice list, where each element is
            a dict with at least:
                - 'image_position': (x, y, z) in patient coords (mm)
            (This is what `read_axial_dicom_series()` already returns.)

        Returns
        -------
        dict
            Same shape as `levels`, with `axial_slice_idx` added.
            If a level's z is outside the axial volume, that level gets
            `axial_slice_idx: None`.

        Notes
        -----
        - Axial slices vary primarily in z. We match using the z component
          of `image_position` (the position of the top-left voxel of each
          axial slice in patient coordinates).
        - "Nearest" is defined as the slice whose `image_position[2]`
          minimizes |z_slice - z_level|. We also flag the level as out of
          range if the level's z is more than half a slice gap beyond
          either end of the volume.
        """
        if not axial_slice_metadata:
            # Nothing to map to; mark every level as out of range.
            for name in levels:
                levels[name]["axial_slice_idx"] = None
            return levels

        # Collect axial slice z positions.
        axial_z = []
        for s in axial_slice_metadata:
            ipp = s.get("image_position")
            if ipp is None or len(ipp) < 3:
                axial_z.append(None)
            else:
                axial_z.append(float(ipp[2]))

        # Range bounds (ignore None entries when computing bounds).
        valid_z = [z for z in axial_z if z is not None]
        if not valid_z:
            for name in levels:
                levels[name]["axial_slice_idx"] = None
            return levels

        z_min = min(valid_z)
        z_max = max(valid_z)
        # Typical inter-slice spacing for tolerance check.
        if len(valid_z) >= 2:
            sorted_z = sorted(valid_z)
            diffs = [abs(sorted_z[i + 1] - sorted_z[i]) for i in range(len(sorted_z) - 1)]
            median_gap = float(np.median(diffs)) if diffs else 0.0
        else:
            median_gap = 0.0

        tolerance = max(median_gap, 1.0)  # at least 1 mm

        # Gap detection threshold: a level is "covered" only if its
        # distance to the nearest slice is less than 1.5x the median
        # inter-slice spacing. This handles clinical lumbar protocols
        # that take small slice groups per IVD with large gaps between
        # groups.
        gap_threshold = max(median_gap * 1.5, 5.0)  # at least 5 mm

        for name, info in levels.items():
            z_level = info["world_xyz"][2]

            # Out-of-range check (allow half a slice gap beyond either end)
            if z_level < z_min - tolerance / 2 or z_level > z_max + tolerance / 2:
                info["axial_slice_idx"] = None
                info["out_of_range"] = True
                info["out_of_range_reason"] = "outside axial volume"
                continue

            # Nearest slice search
            best_idx = None
            best_diff = float("inf")
            for idx, z_slice in enumerate(axial_z):
                if z_slice is None:
                    continue
                d = abs(z_slice - z_level)
                if d < best_diff:
                    best_diff = d
                    best_idx = idx

            # Gap detection: nearest slice too far away
            if best_diff > gap_threshold:
                info["axial_slice_idx"] = None
                info["out_of_range"] = True
                info["out_of_range_reason"] = (
                    f"nearest slice is {best_diff:.1f} mm away "
                    f"(threshold {gap_threshold:.1f} mm) - "
                    "level falls in a gap between axial slice groups"
                )
            else:
                info["axial_slice_idx"] = best_idx
                info["out_of_range"] = False
                info["axial_distance_mm"] = best_diff

        return levels
