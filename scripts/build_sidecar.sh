#!/usr/bin/env bash
# Builds a standalone localforge binary (via PyInstaller) and places it in
# desktop/src-tauri/binaries/ under the name Tauri's "sidecar" mechanism
# expects: <name>-<rust-target-triple>[.exe]. This is what lets the desktop
# app run with no separate CLI install, no Python, and no terminal --
# tauri.conf.json's `bundle.externalBin` bundles whatever's in that folder
# straight into the packaged app, and desktop/src-tauri/src/lib.rs spawns it
# by name via the shell plugin's sidecar() API instead of assuming
# `localforge` is on the end user's PATH.
#
# Must be run on each OS you intend to ship for -- PyInstaller does not
# cross-compile. Run this before `npm run tauri build` in desktop/.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
OUT_DIR="$REPO_ROOT/desktop/src-tauri/binaries"
mkdir -p "$OUT_DIR"

TARGET_TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
if [ -z "$TARGET_TRIPLE" ]; then
  echo "error: couldn't determine the Rust target triple (is rustc installed?)" >&2
  exit 1
fi

EXT=""
case "$TARGET_TRIPLE" in
  *windows*) EXT=".exe" ;;
esac

echo "Building the localforge sidecar for $TARGET_TRIPLE ..."

uv run --with pyinstaller pyinstaller --onefile --noconfirm --name localforge \
  --add-data "src/localforge/catalog_data.yaml:localforge" \
  --collect-data litellm \
  --collect-data certifi \
  --collect-all tiktoken \
  --hidden-import tiktoken_ext.openai_public \
  --collect-submodules tiktoken_ext \
  --hidden-import localforge.backends.ollama \
  --hidden-import localforge.backends.comfyui \
  scripts/pyinstaller_entry.py

DEST="$OUT_DIR/localforge-$TARGET_TRIPLE$EXT"
cp "dist/localforge$EXT" "$DEST"
chmod +x "$DEST"

echo "Sidecar ready: $DEST ($(du -h "$DEST" | cut -f1))"
echo "Sanity check:"
env -i HOME="$HOME" PATH=/usr/bin:/bin "$DEST" --help >/dev/null && echo "  OK: runs standalone with a stripped-down PATH/env."
