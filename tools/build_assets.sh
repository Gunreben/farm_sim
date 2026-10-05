#!/usr/bin/env bash
# Rebuild all third-party-derived assets (not in git: licences / size).
#   STICK=/run/media/$USER/46E7-B98B/Simulation tools/build_assets.sh
# Needs: blender (5.x) on PATH, the Isaac venv (for pxr), internet for Poly Haven (CC0).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STICK="${STICK:-/run/media/$USER/46E7-B98B/Simulation}"
A="$STICK/3D Assets"
PY="${ISAACSIM_VENV:-$HOME/isaacsim/.venv}/bin/python"
TMP="$(mktemp -d)"
cd "$ROOT"

# 1. CC0 skies, ground textures and plants
"$PY" tools/fetch_polyhaven.py

# 2. wheat (Turbosquid pack; the full pack is the .blend1 backup on the stick, textures are packed)
cp "$A/Plants/Crops/Wheat_Pack_Blend_Turbosquid/Wheat_Pack.blend1" "$TMP/wheat.blend"
blender -b "$TMP/wheat.blend" -P tools/blender/export_wheat.py -- assets/crops/wheat
"$PY" tools/postprocess_wheat.py assets/crops/wheat

# 3. actors (stick assets; Sketchfab ones are CC-BY-4.0, licence files copied next to the USD)
C=tools/blender/convert_asset.py
for z in haystack_1 hay_bales; do mkdir -p "$TMP/$z" && unzip -oq "$A/Haybales/$z.zip" -d "$TMP/$z"; done
unzip -oq "$A/Machines/tractor_02/source/New_holland_T7_Tractor_SF.zip" -d "$TMP/nh_t7"
blender -b -P $C -- "$TMP/haystack_1/scene.gltf" assets/actors/round_bale_single/round_bale_single.usd
blender -b -P $C -- "$TMP/hay_bales/scene.gltf" assets/actors/bales_mixed/bales_mixed.usd
cp "$TMP/haystack_1/license.txt" assets/actors/round_bale_single/
cp "$TMP/hay_bales/license.txt" assets/actors/bales_mixed/
blender -b -P $C -- "$TMP/nh_t7/New_holland_T7_Tractor_SF.obj" assets/actors/tractor_trailer/tractor_trailer.usd --size L:9.3
blender -b -P $C -- "$A/Animals/deer.fbx" assets/actors/deer/deer.usd --size H:1.15 --keep-anim --color 0.36,0.22,0.12
blender -b -P $C -- "$A/Animals/cow2/cow/cow.obj" assets/actors/cow/cow.usd --size H:1.45
rm -rf "$TMP"
echo "assets rebuilt"
