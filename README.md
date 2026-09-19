# webcam-depth

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
```

Controls in the live view: `q` quit, `s` save a snapshot into `snapshots/`.

## Model options

| `--encoder` | Params | Notes |
|-------------|--------|-------|
| `vits`      | ~24.8M | ~15-30 FPS, smooth fallback |
| `vitb`      | ~97M   | default, best quality/speed |
| `vitl`      | ~335M  | max quality, use on M2 Pro/Max+ |

Inference uses the Metal Performance Shaders backend (`mps`) when available.

## Tests

```bash
.venv/bin/python -m pytest
```