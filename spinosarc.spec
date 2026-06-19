# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for SpinoSarc.app
Build with: pyinstaller spinosarc.spec  (run from project root)
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).resolve()
SPINOSARC_PKG = PROJECT_ROOT / 'spinosarc_app'
MUSCLEMAP_SCRIPTS = Path.home() / 'SpinoSarc' / 'MuscleMap' / 'scripts'

def find_dcm2niix():
    env = os.environ.get('SPINOSARC_DCM2NIIX_BUILD')
    if env and Path(env).is_file():
        return env
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        candidate = Path(conda_prefix) / 'bin' / 'dcm2niix'
        if candidate.is_file():
            return str(candidate)
    import shutil
    return shutil.which('dcm2niix')

DCM2NIIX_BIN = find_dcm2niix()
if not DCM2NIIX_BIN:
    raise RuntimeError("dcm2niix not found. Activate spinosarc env.")

if not MUSCLEMAP_SCRIPTS.is_dir():
    raise RuntimeError(f"MuscleMap not found at {MUSCLEMAP_SCRIPTS}")
weights_check = MUSCLEMAP_SCRIPTS / 'models' / 'abdomen' / 'v0.0' / 'contrast_agnostic_abdomen_model.pth'
if not weights_check.is_file():
    raise RuntimeError(f"Missing MuscleMap weights at {weights_check}")

print(f"[spec] PROJECT_ROOT      = {PROJECT_ROOT}")
print(f"[spec] MUSCLEMAP_SCRIPTS = {MUSCLEMAP_SCRIPTS}")
print(f"[spec] DCM2NIIX_BIN      = {DCM2NIIX_BIN}")

datas = [
    (str(MUSCLEMAP_SCRIPTS), 'musclemap_scripts'),
]

from PyInstaller.utils.hooks import collect_all

# Module isimlerinin gercek import isimleri:
# pylibjpeg-libjpeg  -> 'libjpeg'
# pylibjpeg-openjpeg -> 'openjpeg'
extra_datas = []
extra_binaries = []
extra_hidden = []
for pkg in ['pylibjpeg', 'libjpeg', 'openjpeg',
            'monai', 'nibabel', 'pydicom', 'skimage', 'reportlab']:
    try:
        d, b, h = collect_all(pkg)
        extra_datas.extend(d)
        extra_binaries.extend(b)
        extra_hidden.extend(h)
        print(f"[spec] collect_all({pkg}): {len(d)} data, {len(b)} bin, {len(h)} hidden")
    except Exception as e:
        print(f"[spec] WARN: collect_all({pkg}) failed: {e}")

datas.extend(extra_datas)

binaries = [(DCM2NIIX_BIN, 'bin')]
binaries.extend(extra_binaries)

hiddenimports = [
    'mm_util',
    'sklearn.cluster', 'sklearn.cluster._kmeans',
    'sklearn.mixture', 'sklearn.mixture._gaussian_mixture',
    'sklearn.utils._cython_blas',
    'sklearn.neighbors', 'sklearn.neighbors.typedefs',
    'sklearn.tree', 'sklearn.tree._utils',
    'scipy.ndimage', 'scipy.special',
    'scipy.special._ufuncs_cxx', 'scipy.special.cython_special',
    'scipy.sparse.csgraph._validation',
    'skimage.filters',
    'monai', 'monai.transforms', 'monai.networks',
    'monai.networks.nets', 'monai.networks.layers',
    'monai.networks.layers.factories', 'monai.inferers',
    'monai.utils', 'monai.utils.module',
    'monai.data', 'monai.config',
    'torch', 'torch._C', 'torch._dynamo',
    'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    'reportlab', 'reportlab.pdfgen', 'reportlab.lib',
    'reportlab.platypus', 'reportlab.graphics',
    'pydicom', 'pydicom.encoders', 'pydicom.encoders.pylibjpeg',
    'pydicom.encoders.gdcm', 'pydicom.pixels', 'pydicom.pixels.decoders',
    # pylibjpeg ve plugin'leri - DOGRU import isimleri:
    'pylibjpeg',
    'libjpeg',         # pylibjpeg-libjpeg paketinin import ismi
    'openjpeg',        # pylibjpeg-openjpeg paketinin import ismi
    'nibabel', 'nibabel.nifti1', 'nibabel.spatialimages',
]
hiddenimports.extend(extra_hidden)
hiddenimports = list(set(hiddenimports))

excludes = [
    'matplotlib.tests', 'numpy.tests', 'scipy.tests',
    'pandas.tests', 'sklearn.tests',
    'tkinter',
    'IPython', 'jupyter', 'notebook',
    'pytest',
    'PyQt6.QtWebEngine', 'PyQt6.QtMultimedia',
]


a = Analysis(
    [str(PROJECT_ROOT / 'spinosarc_launcher.py')],
    pathex=[str(PROJECT_ROOT), str(MUSCLEMAP_SCRIPTS)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='SpinoSarc',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False, argv_emulation=False,
    target_arch='arm64',
    codesign_identity=None, entitlements_file=None,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name='SpinoSarc',
)

app = BUNDLE(
    coll,
    name='SpinoSarc.app',
    icon=None,
    bundle_identifier='com.spinosarc.app',
    version='0.1.0',
    info_plist={
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'CFBundleName': 'SpinoSarc',
        'CFBundleDisplayName': 'SpinoSarc',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHumanReadableCopyright': 'Copyright (c) 2026 Berkay Yilmaz',
        'NSRequiresAquaSystemAppearance': 'False',
        'LSMinimumSystemVersion': '11.0',
    },
)
