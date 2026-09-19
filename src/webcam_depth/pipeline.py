from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

COLORMAP = cv2.COLORMAP_TURBO


def normalize_depth(depth: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    """Percentile-clipped normalization for stable [0, 1] relative depth."""
    clip_lo = float(np.quantile(depth, lo))
    clip_hi = float(np.quantile(depth, hi))
    span = clip_hi - clip_lo
    if span <= 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return np.clip((depth - clip_lo) / span, 0.0, 1.0).astype(np.float32)


def depth_to_bgr(norm: np.ndarray) -> np.ndarray:
    u8 = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, COLORMAP)


class DepthSmoother:
    """Exponential moving average on normalized depth to suppress flicker."""

    def __init__(self, alpha: float = 0.35) -> None:
        self.alpha = float(alpha)
        self._prev: np.ndarray | None = None

    def update(self, norm: np.ndarray) -> np.ndarray:
        if self.alpha <= 0.0 or self._prev is None or self._prev.shape != norm.shape:
            self._prev = norm
        else:
            self._prev = self.alpha * norm + (1.0 - self.alpha) * self._prev
        return self._prev


def _status_line(encoder: str, fps: float, ms: float) -> str:
    return f"{encoder} | {fps:4.1f} FPS | {ms:4.0f} ms | q: quit  s: snapshot"


def run_webcam(
    predictor,
    *,
    source: int = 0,
    smooth_alpha: float = 0.35,
    snap_dir: str | None = "snapshots",
    record_path: str | None = None,
    flip: bool = False,
    width: int = 1280,
    height: int = 720,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {source}")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) if width <= 0 else width
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if height <= 0 else height
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    out = None
    if record_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * int(width), int(height)))

    smoother = DepthSmoother(smooth_alpha)
    snap_path = Path(snap_dir) if snap_dir else None
    if snap_path:
        snap_path.mkdir(parents=True, exist_ok=True)

    fps_ema = 0.0
    frame_idx = 0
    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            depth = predictor.infer(frame)
            norm = normalize_depth(depth)
            smoothed = smoother.update(norm)
            color = depth_to_bgr(smoothed)

            ms = (time.perf_counter() - t0) * 1000.0
            fps_ema = fps_ema * 0.9 + (1000.0 / max(ms, 1.0)) * 0.1

            display = np.hstack([frame, color])
            cv2.putText(
                display,
                _status_line(predictor.encoder, fps_ema, ms),
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("webcam-depth", display)
            if out is not None:
                out.write(display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and snap_path is not None:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                outfile = snap_path / f"depth_{stamp}.png"
                cv2.imwrite(str(outfile), display)
                print(f"[saved] {outfile}")

            frame_idx += 1
    finally:
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()


def run_single_image(predictor, image_path: str, out_dir: str = "snapshots") -> Path:
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Could not read image {image_path}")
    depth = predictor.infer(frame)
    norm = normalize_depth(depth)
    color = depth_to_bgr(norm)
    display = np.hstack([frame, color])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{Path(image_path).stem}_depth.png"
    cv2.imwrite(str(outfile), display)
    print(f"[saved] {outfile}")
    return outfile