#!/bin/bash
set -e
APP="dist/SpinoSarc.app"
VERSION="0.1.0"
DMG_NAME="SpinoSarc-${VERSION}.dmg"
DMG_PATH="dist/${DMG_NAME}"
VOLUME_NAME="SpinoSarc ${VERSION}"

echo "=== SpinoSarc DMG Builder ==="

if [[ ! -d "$APP" ]]; then
    echo "ERROR: $APP yok. Once: ./build_app.sh"
    exit 1
fi

rm -f "$DMG_PATH"
STAGING="dist/_dmg_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"

cat > "$STAGING/INSTALL.txt" << 'README_EOF'
SpinoSarc v0.1.0

Kurulum:
1. SpinoSarc'i Applications klasorune surukle birak.
2. Applications'a git, SpinoSarc'a SAG TIK -> "Ac" -> "Ac"
   (Ilk acilis icin macOS Gatekeeper'i bypass etmek gerekir,
    sonraki acilislarda direkt calisir.)

Sistem Gereksinimleri:
- macOS 11.0 (Big Sur) veya uzeri
- Apple Silicon Mac (M1, M2, M3, M4)
- ~2 GB disk alani

Berkay Yilmaz - Cerrahpasa Tip Fakultesi
README_EOF

echo "DMG olusturuluyor..."
create-dmg \
    --volname "$VOLUME_NAME" \
    --window-pos 200 120 \
    --window-size 700 450 \
    --icon-size 100 \
    --icon "SpinoSarc.app" 175 200 \
    --hide-extension "SpinoSarc.app" \
    --app-drop-link 525 200 \
    --no-internet-enable \
    "$DMG_PATH" \
    "$STAGING" 2>&1 | tail -20

rm -rf "$STAGING"

if [[ -f "$DMG_PATH" ]]; then
    SIZE=$(du -sh "$DMG_PATH" | cut -f1)
    echo ""
    echo "================================"
    echo "  DMG SUCCESS"
    echo "================================"
    echo "  Dosya: $DMG_PATH"
    echo "  Boyut: $SIZE"
    echo "  Test:  open $DMG_PATH"
else
    echo "ERROR: DMG olusturulamadi"
    exit 1
fi
