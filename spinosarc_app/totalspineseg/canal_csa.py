"""
Canal CSA computation from TotalSpineSeg's step1_canal soft mask.

Single responsibility: given a soft canal mask NIfTI and a list of axial
slice world-z positions, compute the cross-sectional area (mm^2) of the
spinal canal at each level.

The canal NIfTI is in sagittal acquisition space (TotalSpineSeg's standard
output). To compute CSA at an axial level, we resample the canal mask at
the world z-coordinate of that axial slice. Because the canal NIfTI is
isotropic 1mm by design (TSS uses iso=True), each "axial" cross-section
in canal-space is a single voxel layer perpendicular to the patient's
superior-inferior axis.

Implementation:
- Locate the voxel axis in the canal NIfTI that maps to patient z (SI).
- For each target axial z (world mm), find the nearest voxel index along
  that axis.
- Slice the canal volume at that index -> 2D soft mask.
- Apply threshold (default 0.5) -> binary mask.
- Count voxels and multiply by in-plane voxel area.

Does NOT:
- Run TotalSpineSeg (runner.py's job)
- Map axial slices to anatomical levels (level_mapper.py's job)
- Touch the GUI

Returns CSA values in mm^2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib


DEFAULT_THRESHOLD = 0.5


def _resolve_si_axis(affine: np.ndarray) -> int:
    """Return the voxel axis (0, 1, or 2) that maps most strongly to world z."""
    return int(np.abs(affine[2, :3]).argmax())


def _resolve_inplane_voxel_area_mm2(affine: np.ndarray, si_axis: int) -> float:
    """In-plane voxel area = product of voxel spacings on the two non-SI axes.

    The spacing of voxel axis k is the L2 norm of the k-th column of the
    3x3 rotation/scaling part of the affine.
    """
    spacings = np.linalg.norm(affine[:3, :3], axis=0)  # one spacing per voxel axis
    in_plane_axes = [a for a in (0, 1, 2) if a != si_axis]
    return float(spacings[in_plane_axes[0]] * spacings[in_plane_axes[1]])


def compute_canal_csa(
    canal_nifti_path: str,
    z_world_mm: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute spinal canal CSA at a single world-z coordinate.

    Parameters
    ----------
    canal_nifti_path
        Path to step1_canal/*.nii.gz from TotalSpineSeg.
    z_world_mm
        Target axial slice z position (patient coords, mm).
    threshold
        Soft mask threshold (default 0.5). Voxels with value > threshold
        are counted as canal.

    Returns
    -------
    dict
        {
            "csa_mm2": float,                # cross-sectional area
            "voxel_count": int,              # voxels above threshold
            "voxel_area_mm2": float,         # in-plane mm^2 per voxel
            "canal_voxel_idx": int,          # voxel index along SI axis
            "z_world_match_mm": float,       # z at the matched voxel
            "z_distance_mm": float,          # |target - matched| mm
        }

    Raises
    ------
    FileNotFoundError
        If the canal NIfTI doesn't exist.
    ValueError
        If the canal NIfTI doesn't cover the requested z (out of volume).
    """
    p = Path(canal_nifti_path)
    if not p.is_file():
        raise FileNotFoundError(f"Canal NIfTI not found: {canal_nifti_path}")

    img = nib.load(str(p))
    data = np.asarray(img.get_fdata())
    affine = np.asarray(img.affine)

    si_axis = _resolve_si_axis(affine)
    si_step = float(affine[2, si_axis])
    si_origin = float(affine[2, 3])
    n_along = data.shape[si_axis]

    # z = si_origin + si_step * voxel_idx  =>  voxel_idx = (z - si_origin) / si_step
    raw_idx = (z_world_mm - si_origin) / si_step
    voxel_idx = int(round(raw_idx))

    if voxel_idx < 0 or voxel_idx >= n_along:
        raise ValueError(
            f"z={z_world_mm:.1f} mm is outside canal NIfTI volume "
            f"(voxel idx {voxel_idx}, range [0, {n_along - 1}])"
        )

    z_matched = si_origin + si_step * voxel_idx
    z_distance = abs(z_world_mm - z_matched)

    # Extract the 2D cross-section perpendicular to SI axis.
    if si_axis == 0:
        cross = data[voxel_idx, :, :]
    elif si_axis == 1:
        cross = data[:, voxel_idx, :]
    else:
        cross = data[:, :, voxel_idx]

    # Threshold + count
    mask = cross > threshold
    voxel_count = int(mask.sum())

    voxel_area = _resolve_inplane_voxel_area_mm2(affine, si_axis)
    csa_mm2 = voxel_count * voxel_area

    return {
        "csa_mm2": float(csa_mm2),
        "voxel_count": voxel_count,
        "voxel_area_mm2": float(voxel_area),
        "canal_voxel_idx": voxel_idx,
        "z_world_match_mm": float(z_matched),
        "z_distance_mm": float(z_distance),
    }


def compute_all_level_csas(
    canal_nifti_path: str,
    levels: dict,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Compute canal CSA for every level in a LevelMapper output dict.

    Parameters
    ----------
    canal_nifti_path
        Path to step1_canal/*.nii.gz.
    levels
        Output of LevelMapper.parse() + map_to_axial(). Each value must
        contain "world_xyz". Levels with axial_slice_idx is None are
        still attempted (canal coverage may differ from axial coverage).
    threshold
        Soft mask threshold.

    Returns
    -------
    dict
        Same keys as `levels`. Each value gets a new key "canal_csa":
            - on success: dict from compute_canal_csa()
            - on failure: {"error": "<message>"}
    """
    out = {}
    for name, info in levels.items():
        try:
            z = float(info["world_xyz"][2])
            result = compute_canal_csa(canal_nifti_path, z, threshold)
            out[name] = {**info, "canal_csa": result}
        except (ValueError, FileNotFoundError) as e:
            out[name] = {**info, "canal_csa": {"error": str(e)}}
    return out


def resample_canal_to_axial_slice(
    canal_nifti_path: str,
    axial_slice_metadata: dict,
    threshold: float = DEFAULT_THRESHOLD,
):
    """Resample the canal soft mask into an axial DICOM slice's pixel grid.

    This is for VISUAL OVERLAY only - it produces a 2D boolean mask aligned
    to the axial slice that the GUI is showing, regardless of orientation
    differences between sagittal canal NIfTI and axial DICOM.

    Parameters
    ----------
    canal_nifti_path
        Path to step1_canal/*.nii.gz.
    axial_slice_metadata
        A single entry from SpinoSarc's axial_slices list. Must contain:
            - 'image_position': (x, y, z) IPP in patient coords (mm)
            - 'image_orientation': 6-element list [row_dir(3), col_dir(3)]
            - 'pixel_spacing': (row_spacing, col_spacing) in mm
            - 'pixel_array': 2D ndarray (used only for target shape)

    Returns
    -------
    np.ndarray or None
        2D boolean mask same shape as axial_slice_metadata['pixel_array'],
        True where canal soft value > threshold. None if anything fails.
    """
    import numpy as _np
    try:
        import nibabel as _nib
        from scipy.ndimage import map_coordinates as _map_coords
    except Exception as e:
        print(f"[CANAL RESAMPLE] import error: {e}")
        return None

    try:
        img = _nib.load(str(canal_nifti_path))
        data = _np.asarray(img.get_fdata())
        affine = _np.asarray(img.affine)
        inv = _np.linalg.inv(affine)

        # Build axial pixel -> patient world transform
        ipp = _np.array(axial_slice_metadata["image_position"], dtype=float)
        iop = _np.array(axial_slice_metadata["image_orientation"], dtype=float)
        row_dir = iop[:3]
        col_dir = iop[3:6]
        ps = axial_slice_metadata.get("pixel_spacing", (1.0, 1.0))
        row_spacing = float(ps[0])
        col_spacing = float(ps[1])

        target_shape = axial_slice_metadata["pixel_array"].shape  # (rows, cols)
        nrow, ncol = target_shape

        # Build a meshgrid of (row, col) -> patient world
        rr, cc = _np.meshgrid(
            _np.arange(nrow, dtype=float),
            _np.arange(ncol, dtype=float),
            indexing="ij",
        )
        # World coords for each pixel - DICOM LPS convention
        # world = IPP + col * col_spacing * row_dir + row * row_spacing * col_dir
        # NOTE: DICOM convention. row_dir is the increment direction for COL index,
        # col_dir is the increment direction for ROW index.
        wx_lps = ipp[0] + cc * col_spacing * row_dir[0] + rr * row_spacing * col_dir[0]
        wy_lps = ipp[1] + cc * col_spacing * row_dir[1] + rr * row_spacing * col_dir[1]
        wz_lps = ipp[2] + cc * col_spacing * row_dir[2] + rr * row_spacing * col_dir[2]

        # Convert DICOM LPS -> NIfTI RAS so we can apply the canal NIfTI's
        # inverse affine (which expects RAS world coordinates).
        # LPS -> RAS: X *= -1, Y *= -1, Z unchanged
        wx = -wx_lps
        wy = -wy_lps
        wz = wz_lps

        # Apply inverse affine: world -> canal voxel
        # Stack as homogeneous: 4 x N
        ones = _np.ones_like(wx)
        world_pts = _np.stack([wx, wy, wz, ones], axis=0).reshape(4, -1)
        canal_vox = inv @ world_pts  # 4 x N
        cvi = canal_vox[0].reshape(nrow, ncol)
        cvj = canal_vox[1].reshape(nrow, ncol)
        cvk = canal_vox[2].reshape(nrow, ncol)

        # Interpolate canal data at these voxel coordinates
        coords = _np.stack([cvi, cvj, cvk], axis=0)
        sampled = _map_coords(data, coords, order=1, mode="constant", cval=0.0)
        mask = sampled > threshold
        return mask
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[CANAL RESAMPLE] error: {e}")
        return None


# ---------------------------------------------------------------------------
# Stenosis classification (literature thresholds)
# ---------------------------------------------------------------------------
# Widely cited dural sac cross-sectional area (DSCA) cutoffs for lumbar
# central canal stenosis (Schonstrom-derived):
#   < 75 mm2  -> absolute stenosis
#   < 100 mm2 -> relative stenosis
#   < 130 mm2 -> early / suspicious
#   >= 130 mm2 -> normal
# NOTE: L5-S is special - the cauda equina has few rootlets there, so a small
# CSA may NOT indicate clinically relevant stenosis. We flag this separately.
# NOTE: supine MRI underestimates stenosis vs axial-loaded/standing.

STENOSIS_ABSOLUTE = 75.0
STENOSIS_RELATIVE = 100.0
STENOSIS_EARLY = 130.0


def classify_stenosis(csa_mm2: float, level_name: str = "") -> dict:
    """Classify a dural sac CSA against literature thresholds.

    This is a NEUTRAL measurement aid, NOT a diagnosis.
    """
    if csa_mm2 < STENOSIS_ABSOLUTE:
        category = "absolute"
        label = f"below absolute threshold (<{STENOSIS_ABSOLUTE:.0f} mm2)"
        flag = True
    elif csa_mm2 < STENOSIS_RELATIVE:
        category = "relative"
        label = f"below relative threshold (<{STENOSIS_RELATIVE:.0f} mm2)"
        flag = True
    elif csa_mm2 < STENOSIS_EARLY:
        category = "early"
        label = f"below early threshold (<{STENOSIS_EARLY:.0f} mm2)"
        flag = True
    else:
        category = "normal"
        label = "normal"
        flag = False

    caveat = None
    if level_name.upper().replace(" ", "") in ("L5-S", "L5-S1", "L5S", "L5S1"):
        caveat = ("L5-S: CSA thresholds are less reliable here "
                  "(few cauda equina rootlets remain).")

    return {
        "category": category,
        "label": label,
        "flag": flag,
        "caveat": caveat,
    }
