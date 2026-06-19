#!/bin/bash
# build_app.sh - SpinoSarc.app builder for Apple Silicon
set -e

echo "=== SpinoSarc Build ==="
echo ""

if [[ -z "$CONDA_PREFIX" ]]; then
    echo "ERROR: Activate spinosarc env first: conda activate spinosarc"
    exit 1
fi

if [[ "$(basename $CONDA_PREFIX)" != "spinosarc" ]]; then
    echo "WARNING: Active env is '$(basename $CONDA_PREFIX)', not 'spinosarc'."
    read -p "Continue anyway? (y/N) " confirm
    [[ "$confirm" != "y" ]] && exit 1
fi

if ! command -v pyinstaller &> /dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

if ! command -v dcm2niix &> /dev/null; then
    echo "ERROR: dcm2niix not found. Install:"
    echo "  conda install -c conda-forge dcm2niix"
    exit 1
fi

# JPEG decompression kontrol - DOGRU import isimleriyle
python3 -c "
import pylibjpeg
import libjpeg
import openjpeg
print(f'pylibjpeg {pylibjpeg.__version__}, libjpeg, openjpeg: OK')
" || {
    echo "ERROR: JPEG decompression eksik. Kur:"
    echo "  pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg"
    exit 1
}

# DICOM decompression gercek testi
python3 -c "
import pydicom
ds = pydicom.dcmread('/Users/berkayyilmaz/Desktop/SynapseMediaSets/Syn20260523233419/DICOMOBJ/00000033', force=True)
arr = ds.pixel_array
print(f'DICOM decompress test: OK ({arr.shape})')
" 2>&1 || echo "WARN: Test DICOM decompress basarisiz (test verisi yoksa normal)"

echo ""
echo "Cleaning previous build..."
rm -rf build/ dist/

echo ""
echo "Running PyInstaller (5-15 minutes)..."
echo "Log: /tmp/spinosarc_build.log"
echo ""
pyinstaller --clean --noconfirm spinosarc.spec 2>&1 | tee /tmp/spinosarc_build.log

echo ""
if [[ -d "dist/SpinoSarc.app" ]]; then
    APP_SIZE=$(du -sh dist/SpinoSarc.app | cut -f1)
    echo "================================"
    echo "  BUILD SUCCESS"
    echo "================================"
    echo "  App:  dist/SpinoSarc.app"
    echo "  Size: $APP_SIZE"
    echo ""
    echo "Test (terminal output icin):"
    echo "  ./dist/SpinoSarc.app/Contents/MacOS/SpinoSarc"
    echo ""
    echo "Veya normal:"
    echo "  open dist/SpinoSarc.app"
else
    echo "================================"
    echo "  BUILD FAILED"
    echo "================================"
    echo "Log: tail -100 /tmp/spinosarc_build.log"
    exit 1
fi
