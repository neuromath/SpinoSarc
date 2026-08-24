"""dicom_loader.py — Synapse PACS DICOM klasörünü scan eder, T2 axial + sagittal
sekansları bulur, dcm2niix ile NIfTI'ye çevirir, multi-station msma axial'i tek
volume'e stackler.

Public API:
    load_synapse_set(dicom_folder, work_dir) -> dict
        keys:
            'axial_nifti'      : Path | None       (stacked axial NIfTI)
            'sagittal_nifti'   : Path | None
            'axial_candidates' : list[dict]        (>1 ise GUI'ye seçim sun)
            'sagittal_candidates': list[dict]
            'info'             : dict              (genel bilgi)
            'errors'           : list[str]

    convert_picked(folder, work_dir, axial_uid, sagittal_uid) -> dict
        kullanıcı dialog'da seçim yaptıktan sonra çağrılır
"""
from __future__ import annotations
import os
import re
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib

try:
    import pydicom
    from pydicom.dataset import FileMetaDataset
    from pydicom.errors import InvalidDicomError
    from pydicom.uid import ImplicitVRLittleEndian, PYDICOM_IMPLEMENTATION_UID
except ImportError as e:
    raise ImportError("pydicom required: pip install pydicom") from e


def align_segmentation_to_frame(segmentation, frame_shape):
    """Return a 2-D segmentation aligned with a decoded DICOM frame.

    ``write_axial_slice_nifti`` writes the decoded frame directly and MONAI's
    inverse transform returns predictions in that original array space.  The
    normal path is therefore identity.  A transpose is accepted only as a
    defensive compatibility path for older inference outputs; arbitrary
    rotations would silently put labels on the wrong anatomy.
    """
    seg = np.asarray(segmentation)
    seg = np.squeeze(seg)
    expected = tuple(int(value) for value in frame_shape)
    if seg.ndim != 2:
        raise ValueError(
            f"Expected a 2-D segmentation, got shape {tuple(seg.shape)}"
        )
    if tuple(seg.shape) == expected:
        return seg
    if tuple(seg.T.shape) == expected:
        return seg.T
    raise ValueError(
        "Segmentation/frame shape mismatch: "
        f"mask={tuple(seg.shape)}, frame={expected}"
    )


# ----- Classification hint dictionaries -----
AXIAL_HINTS = [
    '_TRA', ' TRA', 'TRAN', ' AX ', ' AX_', '_AX_', '_AX ', ' AX-', '-AX-',
    'AXIAL', 'AXIYEL', 'AKSIYEL', 'TRANSVERSE',
]
SAGITTAL_HINTS = [
    '_SAG', ' SAG ', ' SAG_', '_SAG_', 'SAGITTAL', 'SAJITAL',
]
CORONAL_HINTS = [
    '_COR', ' COR ', ' COR_', '_COR_', 'CORONAL', 'KORONAL',
]
LOCALIZER_HINTS = [
    'LOCALIZER', 'LOCALISER', 'LOCALIZ', 'LOCALIS',
    'SURVEY', 'SCOUT',
    '3-PL', '3PL', '3 PL', '3 PLANE', 'TRI-PL', 'TRIPL', 'TRI PL',
    'PILOT', 'REPERE',
    ' LOC ', '_LOC_', '_LOC', ' LOC.', '-LOC-', '-LOC',
]
NON_T2_SEQ = [
    'T1W', 'T1_', 'T1-', ' T1 ', 'STIR', 'TIRM', 'FLAIR',
    'DIFF', 'DWI', 'ADC', 'FFE', 'GRE', 'MERGE',
    'B0', 'B800', 'B1000', 'TRACE', 'EADC',
    'MYELO', 'MIP', 'DIXON', 'T2*',
]


# ============================================================
# 1) DICOM scan + classify
# ============================================================

def _classify_orientation_strict(iop):
    if iop is None or len(iop) < 6:
        return 'unknown'
    row = np.array(iop[:3])
    col = np.array(iop[3:6])
    normal = np.cross(row, col)
    abs_n = np.abs(normal)
    ax = int(abs_n.argmax())
    if abs_n[ax] < 0.9:
        return 'oblique'
    return {0: 'sagittal', 1: 'coronal', 2: 'axial'}[ax]


def _classify_orientation_relaxed(iop):
    if iop is None or len(iop) < 6:
        return 'unknown'
    row = np.array(iop[:3])
    col = np.array(iop[3:6])
    normal = np.cross(row, col)
    abs_n = np.abs(normal)
    ax = int(abs_n.argmax())
    return {0: 'sagittal', 1: 'coronal', 2: 'axial'}[ax]


def _padded(s):
    return ' ' + (s or '').upper() + ' '


def _describe_says(descr, hints):
    if not descr:
        return False
    d = _padded(descr)
    return any(h in d for h in hints)


def _effective_orientation(ds):
    iop = getattr(ds, 'ImageOrientationPatient', None)
    strict = _classify_orientation_strict(iop)
    if strict in ('axial', 'sagittal', 'coronal'):
        return strict
    descr = (str(getattr(ds, 'SeriesDescription', '') or '') + ' ' +
             str(getattr(ds, 'ProtocolName', '') or ''))
    relaxed = _classify_orientation_relaxed(iop)
    if _describe_says(descr, AXIAL_HINTS) and relaxed == 'axial':
        return 'axial'
    if _describe_says(descr, SAGITTAL_HINTS) and relaxed == 'sagittal':
        return 'sagittal'
    if _describe_says(descr, CORONAL_HINTS) and relaxed == 'coronal':
        return 'coronal'
    if _describe_says(descr, AXIAL_HINTS):
        return 'axial'
    if _describe_says(descr, SAGITTAL_HINTS):
        return 'sagittal'
    return strict


def _is_localizer(ds):
    descr = (str(getattr(ds, 'SeriesDescription', '') or '') + ' ' +
             str(getattr(ds, 'ProtocolName', '') or ''))
    d = _padded(descr)
    if any(h in d for h in LOCALIZER_HINTS):
        return True
    image_type = [str(x).upper() for x in getattr(ds, 'ImageType', [])]
    if 'DERIVED' in image_type or 'SECONDARY' in image_type or 'POSDISP' in image_type:
        return True
    return False


def _is_stir(ds):
    ti = getattr(ds, 'InversionTime', None)
    if ti is not None and ti != '':
        try:
            if float(ti) > 0:
                return True
        except (ValueError, TypeError):
            pass
    descr = (str(getattr(ds, 'SeriesDescription', '') or '') + ' ' +
             str(getattr(ds, 'ProtocolName', '') or '')).upper()
    return 'STIR' in descr or 'TIRM' in descr or 'FLAIR' in descr


def _has_contrast(ds):
    cb = str(getattr(ds, 'ContrastBolusAgent', '') or '').strip()
    if cb and cb.lower() not in ('no', 'none', '0'):
        return True
    descr = (str(getattr(ds, 'SeriesDescription', '') or '') + ' ' +
             str(getattr(ds, 'ProtocolName', '') or '')).upper()
    return any(s in descr for s in [' POST', '+C', ' GAD', ' KM', 'KONTRAST', 'POST-'])


def _is_t2(ds):
    descr = (str(getattr(ds, 'SeriesDescription', '') or '') + ' ' +
             str(getattr(ds, 'ProtocolName', '') or '')).upper()
    if any(s in descr for s in NON_T2_SEQ):
        if 'T2' not in descr.replace('T2*', ''):
            return False
    if 'T2' in descr and 'T2*' not in descr:
        return True
    try:
        te = _mr_numeric_value(
            ds, 'EchoTime', 'MREchoSequence', 'EffectiveEchoTime')
        tr = _mr_numeric_value(
            ds, 'RepetitionTime',
            'MRTimingAndRelatedParametersSequence', 'RepetitionTime')
        if te >= 60 and tr > 2000:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _mr_numeric_value(ds, direct_name, sequence_name, nested_name):
    """Read classic or Enhanced-MR numeric metadata without pixel decoding."""
    direct = getattr(ds, direct_name, None)
    if direct not in (None, ''):
        return float(direct)

    containers = [ds]
    shared = getattr(ds, 'SharedFunctionalGroupsSequence', None)
    if shared:
        containers.insert(0, shared[0])
    per_frame = getattr(ds, 'PerFrameFunctionalGroupsSequence', None)
    if per_frame:
        containers.insert(0, per_frame[0])

    for container in containers:
        sequence = getattr(container, sequence_name, None)
        if not sequence:
            continue
        value = getattr(sequence[0], nested_name, None)
        if value not in (None, ''):
            return float(value)
    return 0.0


def scan_dicom_folder(folder):
    """Synapse / generic DICOM klasörünü tara. Return: list of series dicts.

    Her dict:
        uid, series_number, n_slices, description, orientation,
        is_t2, is_stir, has_contrast, matrix, slice_thick, TE, TI,
        files: list[Path]  (bu series'e ait DICOM dosya yolları)
    """
    folder = Path(folder)
    dicom_root = folder / 'DICOMOBJ'
    if not dicom_root.exists():
        dicom_root = folder

    files = sorted([f for f in dicom_root.rglob('*') if f.is_file()])
    files = [f for f in files if 'PDSV' not in str(f)
             and f.name.upper() not in ('DICOMDIR', 'AUTORUN.INF', 'LABEL.TXT')
             and not f.name.endswith(('.lnk', '.exe', '.txt', '.bmp', '.inf', '.ini'))]

    series_dict = defaultdict(list)
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            uid = getattr(ds, 'SeriesInstanceUID', None)
            if uid is None:
                continue
            series_dict[uid].append((f, ds))
        except Exception:
            continue

    rows = []
    for uid, items in series_dict.items():
        ds0 = items[0][1]
        descr = str(getattr(ds0, 'SeriesDescription', '-'))
        if _is_localizer(ds0):
            continue
        n_frames = 0
        for _, item_ds in items:
            try:
                n_frames += max(1, int(getattr(item_ds, 'NumberOfFrames', 1) or 1))
            except (TypeError, ValueError):
                n_frames += 1

        rows.append({
            'uid': uid,
            'series_number': getattr(ds0, 'SeriesNumber', 999),
            # Enhanced MR commonly stores an entire series in one file.
            # Count frames, not just files, so it is not rejected as a
            # one-slice sequence.
            'n_slices': n_frames,
            'description': descr,
            'orientation': _effective_orientation(ds0),
            'is_t2': _is_t2(ds0),
            'is_stir': _is_stir(ds0),
            'has_contrast': _has_contrast(ds0),
            'matrix': f"{getattr(ds0, 'Rows', '?')}x{getattr(ds0, 'Columns', '?')}",
            'slice_thick': getattr(ds0, 'SliceThickness', '-'),
            'TE': getattr(ds0, 'EchoTime', '-'),
            'TI': getattr(ds0, 'InversionTime', '-'),
            'files': [item[0] for item in items],
        })
    return rows


def pick_t2_candidates(rows, orientation):
    """Filter & rank T2 candidates for given orientation."""
    candidates = [r for r in rows
                  if r['orientation'] == orientation
                  and r['is_t2']
                  and not r['is_stir']
                  and not r['has_contrast']
                  and r['n_slices'] >= 10]

    def score(r):
        try:
            m = int(str(r['matrix']).split('x')[0])
        except Exception:
            m = 0
        return (r['n_slices'], m)
    candidates.sort(key=score, reverse=True)
    return candidates


def pick_orientation_candidates(rows, orientation):
    """Fallback list when PACS metadata does not explicitly identify T2.

    These candidates are used only if the strict T2 filter found nothing.
    Ranking prefers non-STIR, non-contrast series with more frames, while the
    GUI still presents multiple plausible series to the radiologist.
    """
    candidates = [
        r for r in rows
        if r['orientation'] == orientation and r['n_slices'] >= 3
    ]

    def score(r):
        try:
            matrix = int(str(r['matrix']).split('x')[0])
        except Exception:
            matrix = 0
        return (
            int(r['is_t2']),
            int(not r['is_stir']),
            int(not r['has_contrast']),
            r['n_slices'],
            matrix,
        )

    candidates.sort(key=score, reverse=True)
    return candidates


# ============================================================
# 2) DICOM -> NIfTI via dcm2niix
# ============================================================

def _resolve_dcm2niix():
    """dcm2niix binary'sini bul. PyInstaller bundle, env var veya PATH.

    Sıra:
    1. PyInstaller bundle: sys._MEIPASS/bin/dcm2niix
    2. SPINOSARC_DCM2NIIX env var (dev-mode override)
    3. shutil.which('dcm2niix') - PATH'te kurulu mu
    """
    # 1) Frozen releases must use the bundled/tested converter.
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        bundled = Path(sys._MEIPASS) / 'bin' / 'dcm2niix'
        if bundled.is_file():
            # Check if bundled binary is executable
            if not os.access(bundled, os.X_OK):
                try:
                    os.chmod(bundled, 0o755)
                except Exception:
                    pass
            return str(bundled)

    # 2) Dev-mode override
    env_path = os.environ.get('SPINOSARC_DCM2NIIX')
    if env_path and Path(env_path).is_file():
        return env_path

    # 3) PATH
    found = shutil.which('dcm2niix')
    if found:
        return found

    return None


def _check_dcm2niix():
    """Raise RuntimeError if dcm2niix not found."""
    path = _resolve_dcm2niix()
    if path is None:
        raise RuntimeError(
            "dcm2niix not found. In dev mode, install with: "
            "conda install -c conda-forge dcm2niix. "
            "In bundled app, this is a packaging bug."
        )
    return path


def _is_part10_dicom(path):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(128)
            return stream.read(4) == b'DICM'
    except OSError:
        return False


def _write_part10_copy(src, dst):
    """Create a uniquely named, standards-compliant DICOM staging copy.

    Some PACS exports omit the 128-byte preamble and ``DICM`` marker. pydicom
    can read those files with ``force=True`` but dcm2niix may report that the
    directory contains no DICOM images. Rewriting only those non-Part-10 files
    makes the hand-off deterministic while preserving the transfer syntax.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _is_part10_dicom(src):
        shutil.copyfile(src, dst)
        return dst

    ds = pydicom.dcmread(str(src), force=True)
    if getattr(ds, 'file_meta', None) is None:
        ds.file_meta = FileMetaDataset()
    file_meta = ds.file_meta

    sop_class = getattr(ds, 'SOPClassUID', None)
    sop_instance = getattr(ds, 'SOPInstanceUID', None)
    if not getattr(file_meta, 'MediaStorageSOPClassUID', None) and sop_class:
        file_meta.MediaStorageSOPClassUID = sop_class
    if not getattr(file_meta, 'MediaStorageSOPInstanceUID', None) and sop_instance:
        file_meta.MediaStorageSOPInstanceUID = sop_instance
    if not getattr(file_meta, 'TransferSyntaxUID', None):
        file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    if not getattr(file_meta, 'ImplementationClassUID', None):
        file_meta.ImplementationClassUID = PYDICOM_IMPLEMENTATION_UID
    ds.preamble = b'\0' * 128
    pydicom.dcmwrite(str(dst), ds, enforce_file_format=True)
    return dst


def _stage_series_files(series, destination):
    """Stage every source with a unique name and fail on partial copies."""
    destination = Path(destination)
    if destination.exists():
        for child in destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    destination.mkdir(parents=True, exist_ok=True)

    staged = []
    failures = []
    for index, src in enumerate(series.get('files', [])):
        target = destination / f'{index:06d}.dcm'
        try:
            staged.append(_write_part10_copy(src, target))
        except Exception as exc:
            failures.append(f'{Path(src).name}: {exc}')

    if failures:
        sample = '; '.join(failures[:3])
        raise RuntimeError(
            f'Could not prepare {len(failures)} of '
            f'{len(series.get("files", []))} DICOM files: {sample}')
    if not staged:
        raise RuntimeError('No DICOM files were copied into the conversion folder')
    return staged


def convert_series_to_nifti(series, work_dir):
    """Verilen series'in DICOM dosyalarını geçici klasöre kopyalayıp dcm2niix
    ile NIfTI'ye çevir. Return: list[Path] of generated .nii.gz files
    (multi-station msma için birden fazla olabilir)."""
    dcm2niix_bin = _check_dcm2niix()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # separate folder per series (dcm2niix scans all DICOMs and splits them
    # into series; we pass a single series so no confusion arises)
    series_input = work_dir / f"series_{series['uid'][-12:]}_input"
    _stage_series_files(series, series_input)

    series_output = work_dir / f"series_{series['uid'][-12:]}_nifti"
    series_output.mkdir(exist_ok=True)
    for f in series_output.glob('*.nii*'):
        try: f.unlink()
        except Exception: pass
    for f in series_output.glob('*.json'):
        try: f.unlink()
        except Exception: pass

    cmd = [
        dcm2niix_bin,
        '-z', 'y',                         # gzip nii.gz
        '-f', '%d_s%s',                    # filename: <description>_s<seriesnum>
        '-o', str(series_output),
        str(series_input),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    nifti_files = sorted(series_output.glob('*.nii.gz'))
    if result.returncode != 0 or not nifti_files:
        # Last-resort conversion is intentionally independent of dcm2niix.
        # It covers valid PACS exports that GDCM/dcm2niix does not recognise.
        fallback = series_output / 'pydicom_fallback.nii.gz'
        try:
            _write_series_nifti(series, fallback)
            nifti_files = [fallback]
        except Exception as fallback_exc:
            raise RuntimeError(
                f"dcm2niix failed (exit {result.returncode}):\n"
                f"stdout: {result.stdout[-500:]}\n"
                f"stderr: {result.stderr[-500:]}\n"
                f"pydicom fallback failed: {fallback_exc}"
            ) from fallback_exc
    return nifti_files


# ============================================================
# 3) Multi-station msma stacking
# ============================================================

def _functional_group_value(ds, frame_group, sequence_name, attribute_name,
                            direct_name=None):
    containers = []
    if frame_group is not None:
        containers.append(frame_group)
    shared = getattr(ds, 'SharedFunctionalGroupsSequence', None)
    if shared:
        containers.append(shared[0])
    for container in containers:
        sequence = getattr(container, sequence_name, None)
        if sequence:
            value = getattr(sequence[0], attribute_name, None)
            if value not in (None, ''):
                return value
    return getattr(ds, direct_name or attribute_name, None)


def _read_dicom_frames(series, cache_dir=None):
    """Decode classic single-frame and Enhanced-MR multi-frame instances."""
    source_files = list(series.get('files', []))
    if cache_dir is not None:
        source_files = _stage_series_files(series, cache_dir)

    frames_data = []
    for file_index, f in enumerate(source_files):
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if getattr(ds, 'file_meta', None) is None:
                ds.file_meta = FileMetaDataset()
            if not getattr(ds.file_meta, 'TransferSyntaxUID', None):
                # A headerless PACS object parsed with force=True has no file
                # meta. Synapse's legacy exports use implicit little endian.
                ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            decoded = np.asarray(ds.pixel_array)
            samples_per_pixel = int(getattr(ds, 'SamplesPerPixel', 1) or 1)
            if decoded.ndim == 2:
                frames = [decoded]
            elif decoded.ndim == 3 and samples_per_pixel == 1:
                frames = [decoded[index] for index in range(decoded.shape[0])]
            else:
                continue

            per_frame = getattr(ds, 'PerFrameFunctionalGroupsSequence', None)
            for frame_index, frame in enumerate(frames):
                fg = (per_frame[frame_index]
                      if per_frame and frame_index < len(per_frame) else None)
                ipp = _functional_group_value(
                    ds, fg, 'PlanePositionSequence', 'ImagePositionPatient')
                iop = _functional_group_value(
                    ds, fg, 'PlaneOrientationSequence', 'ImageOrientationPatient')
                pixel_spacing = _functional_group_value(
                    ds, fg, 'PixelMeasuresSequence', 'PixelSpacing')
                slice_thickness = _functional_group_value(
                    ds, fg, 'PixelMeasuresSequence', 'SliceThickness')
                instance_number = _functional_group_value(
                    ds, fg, 'FrameContentSequence', 'InStackPositionNumber',
                    'InstanceNumber')
                if instance_number in (None, ''):
                    dimension_values = _functional_group_value(
                        ds, fg, 'FrameContentSequence', 'DimensionIndexValues')
                    if dimension_values:
                        instance_number = dimension_values[-1]

                if ipp is None or iop is None or pixel_spacing is None:
                    continue

                slope = _functional_group_value(
                    ds, fg, 'PixelValueTransformationSequence', 'RescaleSlope')
                intercept = _functional_group_value(
                    ds, fg, 'PixelValueTransformationSequence', 'RescaleIntercept')
                slope = float(slope or 1.0)
                intercept = float(intercept or 0.0)
                array = frame.astype(np.float32)
                if slope != 1.0 or intercept != 0.0:
                    array = array * slope + intercept

                try:
                    order = int(instance_number)
                except (TypeError, ValueError):
                    order = file_index * 100000 + frame_index

                frames_data.append({
                    'pixel_array': array,
                    'pixel_spacing': (
                        float(pixel_spacing[0]), float(pixel_spacing[1])),
                    'image_position': tuple(float(value) for value in ipp[:3]),
                    'image_orientation': [float(value) for value in iop[:6]],
                    'instance_number': order,
                    'slice_thickness': float(slice_thickness or 1.0),
                    'rows': int(array.shape[0]),
                    'cols': int(array.shape[1]),
                    'source_file': Path(f).name,
                    'source_path': str(f),
                    'source_frame_index': (
                        frame_index if len(frames) > 1 else None),
                    'z_world_mm': float(ipp[2]),
                })
        except Exception:
            continue

    frames_data.sort(key=lambda item: item['instance_number'])
    return frames_data


def read_axial_dicom_series(series, cache_dir=None):
    """T2 axial series'inin tüm DICOM dosyalarını oku, pixel array + meta çıkar.
    InstanceNumber'a göre sırala (RadiAnt/Synapse davranışı).

    Bu fonksiyon dcm2niix kullanmaz - tüm slice'ları bağımsız 2D olarak korur.
    Farklı oblique açılar (msma multi-station) problem değil çünkü slice'lar
    birleştirilmiyor; her biri kendi affine'iyle bağımsız.

    Return: list of dict, each:
        'pixel_array': np.ndarray (H, W)
        'pixel_spacing': (row_mm, col_mm)
        'image_position': (x, y, z) world mm of slice top-left
        'image_orientation': 6 floats (row_cosines + col_cosines)
        'instance_number': int (radiologist scan order)
        'slice_thickness': float (mm)
        'rows': int, 'cols': int
        'source_file': str (DICOM filename)
        'z_world_mm': float (= image_position[2], cached for convenience)
    """
    return _read_dicom_frames(series, cache_dir=cache_dir)


def _dicom_affine(slice_data, slice_spacing=None):
    """Return a NIfTI RAS affine for a DICOM LPS frame."""
    iop = np.asarray(slice_data['image_orientation'], dtype=float)
    row_direction = iop[:3]
    column_direction = iop[3:6]
    row_spacing, column_spacing = slice_data['pixel_spacing']
    normal = np.cross(row_direction, column_direction)
    normal_norm = np.linalg.norm(normal)
    if normal_norm:
        normal = normal / normal_norm
    spacing = float(slice_spacing or slice_data.get('slice_thickness') or 1.0)

    affine_lps = np.eye(4, dtype=float)
    # NumPy axis 0 advances down image rows; axis 1 advances columns.
    affine_lps[:3, 0] = column_direction * float(row_spacing)
    affine_lps[:3, 1] = row_direction * float(column_spacing)
    affine_lps[:3, 2] = normal * spacing
    affine_lps[:3, 3] = np.asarray(slice_data['image_position'], dtype=float)
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    return lps_to_ras @ affine_lps


def write_axial_slice_nifti(slice_data, output_path):
    """Write one already-decoded DICOM frame without another dcm2niix call."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(slice_data['pixel_array'], dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f'Expected a 2D DICOM frame, got shape {array.shape}')
    image = nib.Nifti1Image(
        array[:, :, np.newaxis], _dicom_affine(slice_data))
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    nib.save(image, str(output_path))
    return output_path


def _write_series_nifti(series, output_path):
    frames = _read_dicom_frames(series)
    if not frames:
        raise RuntimeError('pydicom could not decode any image frames')

    shape = frames[0]['pixel_array'].shape
    frames = [frame for frame in frames if frame['pixel_array'].shape == shape]
    if not frames:
        raise RuntimeError('DICOM frames do not share a common matrix')

    iop = np.asarray(frames[0]['image_orientation'], dtype=float)
    normal = np.cross(iop[:3], iop[3:6])
    normal_norm = np.linalg.norm(normal)
    if normal_norm:
        normal = normal / normal_norm
    frames.sort(key=lambda frame: float(
        np.dot(np.asarray(frame['image_position'], dtype=float), normal)))

    projections = np.asarray([
        np.dot(np.asarray(frame['image_position'], dtype=float), normal)
        for frame in frames
    ])
    differences = np.abs(np.diff(projections))
    differences = differences[differences > 1e-4]
    slice_spacing = (float(np.median(differences)) if differences.size
                     else float(frames[0].get('slice_thickness') or 1.0))
    volume = np.stack([frame['pixel_array'] for frame in frames], axis=2)
    image = nib.Nifti1Image(
        volume.astype(np.float32),
        _dicom_affine(frames[0], slice_spacing=slice_spacing),
    )
    # Keep the DICOM storage axes: axis 2 is the acquired slice direction.
    # The GUI's sagittal viewer and orientation detector intentionally use
    # that convention; canonical RAS reorientation can move the sagittal
    # slice axis to axis 0 and make the viewer scroll through the wrong plane.
    image.set_qform(image.affine, code=1)
    image.set_sform(image.affine, code=1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(output_path))
    return output_path


def _prepare_sagittal(series, work_dir, label='sagittal'):
    """Sagittal series'i NIfTI'ye çevir, tek dosya olarak döndür.
    Sagittal için NIfTI yolu hala kullanılıyor - sagittal genelde tek volume
    (multi-station değil), dcm2niix sorunsuz tek dosya verir."""
    work_dir = Path(work_dir)
    nifti_files = convert_series_to_nifti(series, work_dir)

    if len(nifti_files) == 0:
        return None

    # if multiple fragments, take the largest
    if len(nifti_files) > 1:
        nifti_files.sort(key=lambda f: nib.load(str(f)).shape[2] if nib.load(str(f)).ndim == 3 else 0,
                          reverse=True)

    final = work_dir / f"{label}_s{series['series_number']}.nii.gz"
    if final.exists():
        try: final.unlink()
        except Exception: pass
    shutil.copy2(str(nifti_files[0]), str(final))
    return final


def load_synapse_set(dicom_folder, work_dir=None):
    """Synapse veya generic DICOM klasörünü baştan sona işle.

    Axial: DICOM dosyaları direkt okunur, InstanceNumber'a göre sıralanır
           (RadiAnt/Synapse davranışı). NIfTI dönüşümü YOK - her slice
           kendi affine'iyle bağımsız 2D olarak tutulur.
    Sagittal: dcm2niix ile tek NIfTI (sorun yok, sagittal genelde tek volume).

    Return: dict
        axial_slices: list of dict (her biri bir DICOM slice'ı)
        sagittal_nifti: Path | None
        axial_candidates, sagittal_candidates: list (GUI multi-candidate dialog için)
        info: dict
        errors: list
    """
    folder = Path(dicom_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if work_dir is None:
        work_dir = Path.home() / 'Desktop' / 'SpinoSarc_work' / folder.name
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    result = {
        'axial_slices': [],
        'sagittal_nifti': None,
        'axial_candidates': [],
        'sagittal_candidates': [],
        'selection_required': False,
        'info': {'folder': str(folder), 'work_dir': str(work_dir)},
        'errors': [],
    }

    try:
        rows = scan_dicom_folder(folder)
    except Exception as e:
        result['errors'].append(f"DICOM scan failed: {e}")
        return result

    if not rows:
        result['errors'].append("No DICOM series found in folder")
        return result

    strict_ax_cands = pick_t2_candidates(rows, 'axial')
    strict_sag_cands = pick_t2_candidates(rows, 'sagittal')
    ax_cands = strict_ax_cands or pick_orientation_candidates(rows, 'axial')
    sag_cands = strict_sag_cands or pick_orientation_candidates(rows, 'sagittal')
    result['selection_required'] = bool(
        (not strict_ax_cands and ax_cands)
        or (not strict_sag_cands and sag_cands)
    )

    result['axial_candidates'] = [
        {k: v for k, v in c.items() if k != 'files'} | {'n_files': len(c['files'])}
        for c in ax_cands
    ]
    result['sagittal_candidates'] = [
        {k: v for k, v in c.items() if k != 'files'} | {'n_files': len(c['files'])}
        for c in sag_cands
    ]

    if not strict_ax_cands and ax_cands:
        result['info']['axial_match'] = 'orientation fallback'
    if not strict_sag_cands and sag_cands:
        result['info']['sagittal_match'] = 'orientation fallback'

    if not ax_cands:
        result['errors'].append("No axial T2 candidate found")
    if not sag_cands:
        result['errors'].append("No sagittal T2 candidate found")

    # Single axial candidate: read directly (DICOM mode)
    if len(ax_cands) == 1 and not result['selection_required']:
        try:
            slices = read_axial_dicom_series(
                ax_cands[0], work_dir / 'axial_dicom')
            if not slices:
                result['errors'].append("Axial DICOM read produced no usable slices")
            else:
                result['axial_slices'] = slices
                result['info']['axial_series'] = ax_cands[0]['description']
                result['info']['axial_n_slices'] = len(slices)
        except Exception as e:
            result['errors'].append(f"Axial DICOM read failed: {e}")

    # Single sagittal candidate: convert to NIfTI (legacy path)
    if len(sag_cands) == 1 and not result['selection_required']:
        try:
            result['sagittal_nifti'] = _prepare_sagittal(sag_cands[0], work_dir, 'sagittal')
            result['info']['sagittal_series'] = sag_cands[0]['description']
            result['info']['sagittal_n_slices'] = sag_cands[0]['n_slices']
        except Exception as e:
            result['errors'].append(f"Sagittal conversion failed: {e}")

    return result


def convert_picked(dicom_folder, work_dir, axial_uid, sagittal_uid):
    """Kullanıcı dialog'da seçim yaptıktan sonra çağrılır."""
    folder = Path(dicom_folder)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = scan_dicom_folder(folder)
    result = {'axial_slices': [], 'sagittal_nifti': None, 'errors': []}

    ax = next((r for r in rows if r['uid'] == axial_uid), None)
    sag = next((r for r in rows if r['uid'] == sagittal_uid), None)

    if ax is None:
        result['errors'].append(f"Axial UID not found: {axial_uid}")
    else:
        try:
            slices = read_axial_dicom_series(
                ax, work_dir / 'axial_dicom')
            if not slices:
                result['errors'].append("Axial DICOM read produced no usable slices")
            else:
                result['axial_slices'] = slices
        except Exception as e:
            result['errors'].append(f"Axial DICOM read failed: {e}")

    if sag is None:
        result['errors'].append(f"Sagittal UID not found: {sagittal_uid}")
    else:
        try:
            result['sagittal_nifti'] = _prepare_sagittal(sag, work_dir, 'sagittal')
        except Exception as e:
            result['errors'].append(f"Sagittal conversion failed: {e}")

    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python dicom_loader.py /path/to/Synapse_folder")
        sys.exit(1)
    out = load_synapse_set(sys.argv[1])
    summary = {
        'n_axial_slices': len(out['axial_slices']),
        'first_3_axial': [
            {'instance_number': s['instance_number'],
             'z': s['z_world_mm'],
             'shape': (s['rows'], s['cols']),
             'source': s['source_file']}
            for s in out['axial_slices'][:3]
        ] if out['axial_slices'] else [],
        'last_3_axial': [
            {'instance_number': s['instance_number'],
             'z': s['z_world_mm'],
             'shape': (s['rows'], s['cols']),
             'source': s['source_file']}
            for s in out['axial_slices'][-3:]
        ] if out['axial_slices'] else [],
        'sagittal_nifti': str(out['sagittal_nifti']) if out['sagittal_nifti'] else None,
        'axial_candidates': out['axial_candidates'],
        'sagittal_candidates': out['sagittal_candidates'],
        'info': out['info'],
        'errors': out['errors'],
    }
    print(json.dumps(summary, indent=2, default=str))
