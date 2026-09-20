#!/usr/bin/env bash
# Scan every image in ./images into scans/ with scan-depth, then point at the viewer.
#
# Usage:  ./scan_images.sh                     # auto voxel per image
#         ./scan_images.sh --voxel 0.03        # fixed downsample size

set -euo pipefail
cd "$(dirname "$0")"

SCAN=".venv/bin/scan-depth"
if [[ ! -x "$SCAN" ]]; then
  echo "[error] missing .venv; install with: .venv/bin/python -m pip install -e ." >&2
  exit 1
fi

VOXEL_ARGS=()
if [[ "${1:-}" == "--voxel" ]]; then
  VOXEL_ARGS=(--voxel "${2:?missing voxel size}")
  shift 2
fi

mkdir -p scans

mapfile -t IMAGES < <(find images -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | sort)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "[error] no images found in ./images" >&2
  exit 1
fi

echo "[scan_images] scanning ${#IMAGES[@]} image(s)"
for img in "${IMAGES[@]}"; do
  out="scans/$(basename "${img%.*}").ply"
  echo "==> $img -> $out"
  "$SCAN" --image "$img" -o "$out" "${VOXEL_ARGS[@]}"
done

echo
echo "[scan_images] done. view the results with:"
echo "  .venv/bin/python -m http.server 8000   # then open http://localhost:8000/viewer/"