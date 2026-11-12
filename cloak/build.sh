#!/usr/bin/env bash
# Full, deterministic build of the "Weight of Forgotten Oaths" cloak:
# tatter masks -> geometry + bakes -> PBR textures -> Roblox exports -> preview renders.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
BLENDER="${BLENDER:-/opt/blender-4.5.16-linux-x64/blender}"
BUILD="$ROOT/build"
mkdir -p "$BUILD"

echo "== 1/5 tatter masks"
python3 "$SRC/make_masks.py" "$BUILD"

echo "== 2/5 geometry, hardware, occlusion bakes"
"$BLENDER" -b --factory-startup -P "$SRC/build_scene.py" -- "$BUILD" 2>&1 | grep -E "GROUP|hardware:|bakes done|built|Error|Traceback" || true

echo "== 3/5 textures"
python3 "$SRC/make_textures.py" "$BUILD" "$ROOT/textures"

echo "== 4/5 export"
"$BLENDER" -b --factory-startup "$BUILD/cloak_scene.blend" -P "$SRC/export.py" -- "$ROOT/export" "$ROOT/textures/roblox_1024" \
  2>&1 | grep -E "tris=|export done|Error|Traceback" || true
mkdir -p "$ROOT/roblox"
python3 "$SRC/make_roblox_script.py" "$ROOT/export" "$ROOT/roblox/ForgottenOathsCloakSetup.lua"
rm -f "$ROOT/export/"*.blend1
"$BLENDER" -b --factory-startup -P "$SRC/verify_export.py" -- "$ROOT/export" "$BUILD/verify" 2>&1 | grep -E "VERIFY|MISMATCH|MISSING" || true

echo "== 5/5 renders"
export CLOAK_TEX="$ROOT/textures/roblox_1024"
"$BLENDER" -b --factory-startup "$BUILD/cloak_scene.blend" -P "$SRC/render_views.py" -- \
  "$ROOT/renders/views" final "${VIEW_W:-720}" "${VIEW_H:-960}" "${SAMPLES:-64}" front back left right back34 front34 \
  2>&1 | grep -E "Error|Traceback" || true
"$BLENDER" -b --factory-startup "$BUILD/cloak_scene.blend" -P "$SRC/render_views.py" -- \
  "$ROOT/renders/views" final "${DETAIL_W:-640}" "${DETAIL_W:-640}" "${SAMPLES:-64}" brooch buckle rune hem chest upperback \
  2>&1 | grep -E "Error|Traceback" || true
python3 "$SRC/compose_sheet.py" "$ROOT/renders/views" "$ROOT/renders/ForgottenOathsCloak_sheet.png"
python3 "$SRC/compare_ref.py" "$ROOT/renders/views" "$ROOT/renders/reference_comparison.png"
echo "done"
