#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$ROOT_DIR/dist/SpinoSarc.app"
VERSION="${SPINOSARC_VERSION:-0.3.0}"
DMG_PATH="$ROOT_DIR/dist/SpinoSarc-${VERSION}-Apple-Silicon.dmg"
STAGING="$ROOT_DIR/dist/dmg-staging"

[[ -d "$APP" ]] || { echo "ERROR: Run ./build_app.sh first."; exit 1; }
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
cp "$ROOT_DIR/INSTALL.md" "$STAGING/Read Me.md"

rm -f "$DMG_PATH"
if command -v create-dmg >/dev/null; then
    create-dmg \
        --volname "SpinoSarc $VERSION" \
        --window-size 700 450 \
        --icon-size 100 \
        --icon "SpinoSarc.app" 175 200 \
        --hide-extension "SpinoSarc.app" \
        --app-drop-link 525 200 \
        --no-internet-enable \
        "$DMG_PATH" "$STAGING"
else
    hdiutil create -volname "SpinoSarc $VERSION" -srcfolder "$STAGING" \
        -ov -format UDZO "$DMG_PATH"
fi
rm -rf "$STAGING"

if [[ -n "${SPINOSARC_CODESIGN_IDENTITY:-}" ]]; then
    codesign --force --timestamp --sign "$SPINOSARC_CODESIGN_IDENTITY" "$DMG_PATH"
fi
shasum -a 256 "$DMG_PATH" > "$DMG_PATH.sha256"
echo "Built $DMG_PATH"
