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
    from pydicom.errors import InvalidDicomError
except ImportError as e:
    raise ImportError("pydicom required: pip install pydicom") from e


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
        te = float(getattr(ds, 'EchoTime', 0) or 0)
        tr = float(getattr(ds, 'RepetitionTime', 0) or 0)
        if te >= 60 and tr > 2000:
            return True
    except (ValueError, TypeError):
        pass
    return False


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
        rows.append({
            'uid': uid,
            'series_number': getattr(ds0, 'SeriesNumber', 999),
            'n_slices': len(items),
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


# ============================================================
# 2) DICOM -> NIfTI via dcm2niix
# ============================================================

def _resolve_dcm2niix():
    """dcm2niix binary'sini bul. PyInstaller bundle, env var veya PATH.

    Sıra:
    1. SPINOSARC_DCM2NIIX env var (manuel override)
    2. PyInstaller bundle: sys._MEIPASS/bin/dcm2niix
    3. shutil.which('dcm2niix') - PATH'te kurulu mu
    """
    # 1) Env var
    env_path = os.environ.get('SPINOSARC_DCM2NIIX')
    if env_path and Path(env_path).is_file():
        return env_path

    # 2) PyInstaller bundle
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
    series_input.mkdir(exist_ok=True)
    # clear previous outputs (idempotent operation)
    for f in series_input.glob('*'):
        try: f.unlink()
        except Exception: pass

    for src in series['files']:
        try:
            shutil.copy2(str(src), str(series_input / src.name))
        except Exception:
            pass

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
    if result.returncode != 0:
        raise RuntimeError(
            f"dcm2niix failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )

    nifti_files = sorted(series_output.glob('*.nii.gz'))
    if not nifti_files:
        raise RuntimeError(f"dcm2niix produced no NIfTI files. stdout:\n{result.stdout[-500:]}")
    return nifti_files


# ============================================================
# 3) Multi-station msma stacking
# ============================================================

def read_axial_dicom_series(series):
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
    slices_data = []
    for f in series['files']:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            # pixel_array - PixelData yoksa atla
            try:
                arr = ds.pixel_array
            except Exception:
                continue
            # rescale (uygulanabilirse)
            rescale_slope = float(getattr(ds, 'RescaleSlope', 1.0) or 1.0)
            rescale_intercept = float(getattr(ds, 'RescaleIntercept', 0.0) or 0.0)
            if rescale_slope != 1.0 or rescale_intercept != 0.0:
                arr = arr.astype(np.float32) * rescale_slope + rescale_intercept
            else:
                arr = arr.astype(np.float32)

            ipp = getattr(ds, 'ImagePositionPatient', None)
            iop = getattr(ds, 'ImageOrientationPatient', None)
            ps = getattr(ds, 'PixelSpacing', None)
            inum = getattr(ds, 'InstanceNumber', None)
            sl_thick = getattr(ds, 'SliceThickness', None)

            if ipp is None or iop is None or ps is None:
                continue  # yetersiz metadata

            slices_data.append({
                'pixel_array': arr,
                'pixel_spacing': (float(ps[0]), float(ps[1])),
                'image_position': (float(ipp[0]), float(ipp[1]), float(ipp[2])),
                'image_orientation': [float(x) for x in iop],
                'instance_number': int(inum) if inum is not None else 0,
                'slice_thickness': float(sl_thick) if sl_thick is not None else 1.0,
                'rows': int(arr.shape[0]),
                'cols': int(arr.shape[1]),
                'source_file': Path(f).name,
                'source_path': str(f),
                'z_world_mm': float(ipp[2]),
            })
        except Exception as e:
            # this slice could not be read, continue
            continue

    if not slices_data:
        return []

    # sort by InstanceNumber (RadiAnt-like behavior)
    slices_data.sort(key=lambda s: s['instance_number'])
    return slices_data


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

    ax_cands = pick_t2_candidates(rows, 'axial')
    sag_cands = pick_t2_candidates(rows, 'sagittal')

    result['axial_candidates'] = [
        {k: v for k, v in c.items() if k != 'files'} | {'n_files': len(c['files'])}
        for c in ax_cands
    ]
    result['sagittal_candidates'] = [
        {k: v for k, v in c.items() if k != 'files'} | {'n_files': len(c['files'])}
        for c in sag_cands
    ]

    if not ax_cands:
        result['errors'].append("No axial T2 candidate found")
    if not sag_cands:
        result['errors'].append("No sagittal T2 candidate found")

    # Single axial candidate: read directly (DICOM mode)
    if len(ax_cands) == 1:
        try:
            slices = read_axial_dicom_series(ax_cands[0])
            if not slices:
                result['errors'].append("Axial DICOM read produced no usable slices")
            else:
                result['axial_slices'] = slices
                result['info']['axial_series'] = ax_cands[0]['description']
                result['info']['axial_n_slices'] = len(slices)
        except Exception as e:
            result['errors'].append(f"Axial DICOM read failed: {e}")

    # Single sagittal candidate: convert to NIfTI (legacy path)
    if len(sag_cands) == 1:
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
            slices = read_axial_dicom_series(ax)
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
