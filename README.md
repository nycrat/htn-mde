<img width="2036" height="1271" alt="image" src="https://github.com/user-attachments/assets/5cf5cba4-6aad-4d77-80a6-e777b5ea0a27" />

# MDE Gallery

Three.js WebGL viewer for 3D scan. Visit the website [here](https://nycrat.dev/htn-mde/viewer/).

# MDE CLI

Real-time monocular depth estimation from a webcam, optimized for Apple Silicon (MPS).
Built on [Depth Anything V2](https://github.com/depthanything/Depth-Anything-V2), loaded
through HuggingFace `transformers`.

```
RGB  |  depth
[src]|          <- live side-by-side view, TURBO colormap, FPS + inference ms overlay
```

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # or: .venv/bin/pip install -e . pytest
```

First run downloads the model weights (~100-350 MB) into your HuggingFace cache.

## Usage

```bash
.venv/bin/webcam-depth                     # live webcam, vitb at 518px (quality)
.venv/bin/webcam-depth --encoder vits      # lighter/faster model
.venv/bin/webcam-depth --encoder vitl      # max quality (slowest)
.venv/bin/webcam-depth --input-size 616    # finer detail, slower
.venv/bin/webcam-depth --image photo.jpg   # one-shot depth map, no camera needed
.venv/bin/webcam-depth --record out.mp4    # save the processed stream
.venv/bin/webcam-depth --probe-cameras     # list available camera devices + indexes
```

Controls in the live view: `q` quit, `s` save a snapshot into `snapshots/`.

Camera selection: pass `--source N` to use device `N`. On macOS, `--probe-cameras`
lists what OpenCV can see — this includes the built-in webcam and, when enabled
via **Continuity Camera**, your iPhone (wireless or USB). Frame grab on the
Continuity device can be flaky with older OpenCV wheels; if it fails, the most
reliable path is to shoot stills in the iPhone Camera app, AirDrop them into the
session's `images/` folder, and run `--sfm` on the session.

## Model options

| `--encoder` | Params | Notes                           |
| ----------- | ------ | ------------------------------- |
| `vits`      | ~24.8M | ~15-30 FPS, smooth fallback     |
| `vitb`      | ~97M   | default, best quality/speed     |
| `vitl`      | ~335M  | max quality, use on M2 Pro/Max+ |

Inference uses the Metal Performance Shaders backend (`mps`) when available.

## Tests

```bash
.venv/bin/python -m pytest
```

## 3D scan (colored point cloud)

Emit a colored `.ply` point cloud from mono depth, then view it in the Three.js WebGL viewer.

```bash
# single image -> scan.ply
.venv/bin/scan-depth --image photo.jpg -o scan.ply

# live webcam: press 'c' in the preview to capture a frame
.venv/bin/scan-depth -o scan.ply

# spin sweep: rotate the camera ~360deg around a centered subject
.venv/bin/scan-depth --sweep 40 -o scan.ply
```

Tuning: `--fov 60` (camera horizontal FOV), `--stride 1` (full detail), `--keep 0.01 0.99`
(depth quantile range), `--encoder vits` (faster capture). Scans are voxel-downsampled
by default (`--voxel 0` = auto, ~diag/1000) to keep the point cloud light in the viewer;
pass a larger `--voxel` for an even smaller cloud.

View it. Serve the viewer from the repo root, then open the scans gallery:

```bash
.venv/bin/python -m http.server 8000
open http://localhost:8000/viewer/
```

`.ply` files in `scans/` appear as a thumbnail gallery — click one to load it. The
viewer must be served over `http://` (not opened as a `file://` double-click), since
it fetches the gallery listing and scan files. Drag to orbit, scroll to zoom, near/far
sliders slice depth layers.

Note: webcam depth is relative — the point cloud is correctly shaped but
arbitrarily scaled. The sweep mode uses the turntable assumption (camera spinning
around a centered subject), not full pose estimation.

## Free-motion 3D scan (SfM)

Instead of the turntable sweep, you can move the camera freely around a
stationary, textured object. This uses [COLMAP](https://colmap.github.io/) for
offline pose estimation, then aligns each frame's Depth-Anything cloud to the
recovered poses. COLMAP runs in-process via pip:

```bash
pip install pycolmap      # primary backend (in-process COLMAP)
# alternative: brew install colmap   # the standalone `colmap` binary
```

Two steps, or one command:

```bash
# 1. capture a session (live preview; press 'c' to add a frame, 'q' to finish)
.venv/bin/scan-depth --capture my_object

# 2. estimate poses (COLMAP) + fuse depth into a colored cloud
.venv/bin/scan-depth --sfm my_object -o scan.ply
```

Or in one shot, capture then immediately fuse:

```bash
.venv/bin/scan-depth --capture my_object && .venv/bin/scan-depth --sfm my_object -o scan.ply
```

Capture tips (this matters most for quality):

- keep the camera moving slowly with ~60–80% overlap between consecutive frames;
  the tool warns when consecutive captures differ too much
- aim for 15–30 frames covering the full surface
- the object must be static, textured, and shadow-free in even lighting — blank
  walls and fuzzy or glasslike surfaces will make COLMAP fail
- avoid large motions; rotate or translate smoothly around the subject

`--sfm` reads a `depth/` folder next to the images if present (written by
`--capture`), otherwise it re-runs Depth-Anything per frame. Tuning: `--voxel`
(downsample size, `<=0` = auto), `--stride`, `--keep`, `--bg-ratio` (cull each
frame's pixels farther than `R` x its median SfM-tracked depth; Depth-Anything's
relative depth has an unbounded far end, `0` disables), `--sfm-backend`
(`auto`/`pycolmap`/`colmap`), `--colmap-bin`.
