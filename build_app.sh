#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/.build"
VENV_DIR="$BUILD_DIR/venv"
MUSCLEMAP_DIR="$BUILD_DIR/vendor/MuscleMap"
TSS_DATA_DIR="$BUILD_DIR/totalspineseg_data"
MUSCLEMAP_COMMIT="d11df779a4e146e7e913b89cb202fc06a6e2ef6a"
APP_VERSION="${SPINOSARC_VERSION:-0.3.1}"
PYTHON_BIN="${SPINOSARC_PYTHON:-python3}"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: Apple Silicon (arm64) is required for this release build."
    exit 1
fi

command -v "$PYTHON_BIN" >/dev/null || {
    echo "ERROR: Python 3.11 is required. Set SPINOSARC_PYTHON if needed."
    exit 1
}
command -v git >/dev/null || { echo "ERROR: git is required."; exit 1; }
command -v dcm2niix >/dev/null || {
    echo "ERROR: dcm2niix is required (brew install dcm2niix)."
    exit 1
}

mkdir -p "$BUILD_DIR/vendor"
if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
export ARCHFLAGS="-arch arm64"
export CMAKE_POLICY_VERSION_MINIMUM="3.5"
export MACOSX_DEPLOYMENT_TARGET="11.0"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r "$ROOT_DIR/requirements-macos-build.txt"

if [[ ! -d "$MUSCLEMAP_DIR/.git" ]]; then
    git clone https://github.com/MuscleMap/MuscleMap.git "$MUSCLEMAP_DIR"
fi
git -C "$MUSCLEMAP_DIR" fetch --depth 1 origin "$MUSCLEMAP_COMMIT"
git -C "$MUSCLEMAP_DIR" checkout --detach "$MUSCLEMAP_COMMIT"

PYTHONPATH="$MUSCLEMAP_DIR/scripts" python -c \
    "from mm_util import ensure_model_downloaded; ensure_model_downloaded('abdomen', 'latest')"

mkdir -p "$TSS_DATA_DIR"
python -m totalspineseg.init_inference \
    --data-dir "$TSS_DATA_DIR" --store-export --quiet

export SPINOSARC_VERSION="$APP_VERSION"
export SPINOSARC_MUSCLEMAP_BUILD="$MUSCLEMAP_DIR/scripts"
export SPINOSARC_TSS_DATA_BUILD="$TSS_DATA_DIR"
export SPINOSARC_DCM2NIIX_BUILD="$(command -v dcm2niix)"

rm -rf "$ROOT_DIR/build" "$ROOT_DIR/dist"
python -m PyInstaller --clean --noconfirm "$ROOT_DIR/spinosarc.spec"

APP="$ROOT_DIR/dist/SpinoSarc.app"
if [[ ! -d "$APP" ]]; then
    echo "ERROR: PyInstaller did not create $APP"
    exit 1
fi

if [[ -n "${SPINOSARC_CODESIGN_IDENTITY:-}" ]]; then
    codesign --force --deep --options runtime --timestamp \
        --sign "$SPINOSARC_CODESIGN_IDENTITY" "$APP"
else
    codesign --force --deep --sign - "$APP"
fi

codesign --verify --deep --strict --verbose=2 "$APP"

APP_EXEC="$APP/Contents/MacOS/SpinoSarc"
LOG_PATH="$HOME/Library/Logs/SpinoSarc/SpinoSarc.log"

run_release_check() {
    local check_flag="$1"
    if ! "$APP_EXEC" "$check_flag"; then
        echo "ERROR: Frozen release check failed: $check_flag"
        if [[ -f "$LOG_PATH" ]]; then
            tail -n 200 "$LOG_PATH"
        fi
        exit 1
    fi
}

# Import every user-facing runtime, execute the bundled dcm2niix binary,
# deserialize TotalSpineSeg's actual step-1 checkpoint, and perform a real
# MuscleMap CPU forward pass.  A DMG is never produced if any check fails.
run_release_check --spinosarc-runtime-preflight
run_release_check --spinosarc-tss-preflight
run_release_check --spinosarc-musclemap-self-test

echo "Built $APP"
du -sh "$APP"
