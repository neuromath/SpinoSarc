# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller recipe for the self-contained Apple Silicon application."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, copy_metadata


PROJECT_ROOT = Path(SPECPATH).resolve()
APP_VERSION = os.environ.get("SPINOSARC_VERSION", "0.3.1")
MUSCLEMAP_SCRIPTS = Path(
    os.environ.get(
        "SPINOSARC_MUSCLEMAP_BUILD",
        str(PROJECT_ROOT / ".build" / "vendor" / "MuscleMap" / "scripts"),
    )
)
TSS_DATA = Path(
    os.environ.get(
        "SPINOSARC_TSS_DATA_BUILD",
        str(PROJECT_ROOT / ".build" / "totalspineseg_data"),
    )
)


def find_dcm2niix():
    configured = os.environ.get("SPINOSARC_DCM2NIIX_BUILD")
    if configured and Path(configured).is_file():
        return configured
    import shutil

    return shutil.which("dcm2niix")


DCM2NIIX_BIN = find_dcm2niix()
if not DCM2NIIX_BIN:
    raise RuntimeError("dcm2niix was not found; run build_app.sh")
if not (MUSCLEMAP_SCRIPTS / "mm_util.py").is_file():
    raise RuntimeError(f"MuscleMap scripts not found at {MUSCLEMAP_SCRIPTS}")
if not list((MUSCLEMAP_SCRIPTS / "models").rglob("*.pth")):
    raise RuntimeError("MuscleMap weights are missing; run build_app.sh")
if not list((MUSCLEMAP_SCRIPTS / "models").rglob("*.json")):
    raise RuntimeError("MuscleMap model config is missing; run build_app.sh")
if not list(TSS_DATA.rglob("*.pth")):
    raise RuntimeError("TotalSpineSeg weights are missing; run build_app.sh")


datas = [
    (str(MUSCLEMAP_SCRIPTS), "musclemap_scripts"),
    (str(TSS_DATA), "totalspineseg_data"),
]
binaries = [(DCM2NIIX_BIN, "bin")]
hiddenimports = ["mm_util"]

# TotalSpineSeg and nnU-Net rely on plugin-style/dynamic imports.  Collecting
# their package data and submodules at build time makes the release independent
# of Python, pip, Conda, and the network on the radiologist's Mac.
packages = [
    "pylibjpeg", "libjpeg", "openjpeg", "monai", "nibabel", "pydicom",
    "skimage", "reportlab", "openpyxl", "totalspineseg", "nnunetv2", "auglab",
    "batchgenerators", "dynamic_network_architectures", "acvl_utils",
    "torchio", "nilearn", "gryds",
]
for package in packages:
    try:
        package_data, package_binaries, package_hidden = collect_all(package)
        datas.extend(package_data)
        binaries.extend(package_binaries)
        hiddenimports.extend(package_hidden)
    except Exception as exc:
        print(f"[spec] collect_all({package}) warning: {exc}")

for distribution in (
    "totalspineseg", "nnunetv2", "dynamic-network-architectures",
    "batchgenerators", "torchio", "nilearn", "monai",
):
    try:
        datas.extend(copy_metadata(distribution, recursive=True))
    except Exception as exc:
        print(f"[spec] metadata({distribution}) warning: {exc}")

hiddenimports.extend([
    "sklearn.cluster", "sklearn.cluster._kmeans", "sklearn.mixture",
    "sklearn.mixture._gaussian_mixture", "sklearn.utils._cython_blas",
    "sklearn.neighbors", "sklearn.tree", "sklearn.tree._utils",
    "scipy.ndimage", "scipy.special", "scipy.special._ufuncs_cxx",
    "scipy.special.cython_special", "scipy.sparse.csgraph._validation",
    "monai.transforms", "monai.networks", "monai.inferers", "monai.data",
    "torch", "torch._C", "torch._dynamo", "PyQt6.QtCore", "PyQt6.QtGui",
    "PyQt6.QtWidgets", "pydicom.pixels", "pydicom.pixels.decoders",
    "nibabel.nifti1", "nibabel.spatialimages",
])
hiddenimports = sorted(set(hiddenimports))

excludes = [
    "matplotlib.tests", "numpy.tests", "scipy.tests", "pandas.tests",
    "sklearn.tests", "tkinter", "IPython", "jupyter", "notebook", "pytest",
    "PyQt6.QtWebEngine", "PyQt6.QtMultimedia",
]

a = Analysis(
    [str(PROJECT_ROOT / "spinosarc_launcher.py")],
    pathex=[str(PROJECT_ROOT), str(MUSCLEMAP_SCRIPTS)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SpinoSarc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    argv_emulation=False,
    target_arch="arm64",
    codesign_identity=os.environ.get("SPINOSARC_CODESIGN_IDENTITY") or None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="SpinoSarc",
)
app = BUNDLE(
    coll,
    name="SpinoSarc.app",
    icon=os.environ.get("SPINOSARC_ICON") or None,
    bundle_identifier="org.neuromath.SpinoSarc",
    version=APP_VERSION,
    info_plist={
        "NSPrincipalClass": "NSApplication",
        "NSHighResolutionCapable": True,
        "CFBundleName": "SpinoSarc",
        "CFBundleDisplayName": "SpinoSarc",
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "NSHumanReadableCopyright": "Copyright © 2026 Berkay Yılmaz",
        "NSRequiresAquaSystemAppearance": False,
        "LSMinimumSystemVersion": "14.0",
        "LSArchitecturePriority": ["arm64"],
    },
)
