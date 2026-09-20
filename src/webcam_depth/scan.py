from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pipeline import depth_to_bgr, normalize_depth
from .predictor import DepthPredictor
from .sfm import capture_session, fuse_session, run_colmap, voxel_downsample


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @staticmethod
    def from_fov(width: int, height: int, fov_deg: float = 60.0) -> CameraIntrinsics:
        f = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
        return CameraIntrinsics(f, f, width / 2.0, height / 2.0)


def probe_cameras(max_index: int = 10) -> list[dict[str, int | float | str]]:
    """List camera device indices this OpenCV build can open (macOS: includes
    Continuity Camera iPhones). Each probe grabs a frame and reports size/FPS."""
    desc_prop = getattr(cv2, "CAP_PROP_DEVICE_DESCRIPTION", None)
    found: list[dict[str, int | float | str]] = []
    misses = 0
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            misses += 1
            if misses >= 3:
                break
            continue
        misses = 0
        gave_frame = False
        ok, frame = cap.read()
        gave_frame = ok and frame is not None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        name = ""
        if desc_prop is not None:
            try:
                name = str(cap.get(desc_prop))
            except cv2.error:
                name = ""
        if not name:
            name = "AVFoundation device"
        cap.release()
        found.append(
            {
                "index": idx,
                "name": name,
                "width": w,
                "height": h,
                "fps": fps,
                "frame": gave_frame,
            }
        )
    return found


def unproject(
    frame_bgr: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    stride: int = 2,
    keep: tuple[float, float] = (0.01, 0.99),
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a BGR frame + depth map into a colored point cloud.

    Returns (points, rgb) with points as (N, 3) float32 and rgb as (N, 3) uint8.
    Depth is treated as distance along the camera +Z axis (forward). Relative
    depth is fine: the result is a correctly-shaped, arbitrarily-scaled model.
    """
    h, w = depth.shape[:2]
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    keep_lo, keep_hi = np.quantile(finite, float(keep[0])), np.quantile(finite, float(keep[1]))
    valid = np.isfinite(depth) & (depth >= keep_lo) & (depth <= keep_hi)

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    sel = valid[::stride, ::stride]
    xs = xs[sel].astype(np.float32)
    ys = ys[sel].astype(np.float32)
    d = depth[::stride, ::stride][sel].astype(np.float32)

    pts = np.empty((int(sel.sum()), 3), dtype=np.float32)
    pts[:, 0] = (xs - intrinsics.cx) * d / intrinsics.fx
    pts[:, 1] = -(ys - intrinsics.cy) * d / intrinsics.fy  # image v grows down -> +Y up
    pts[:, 2] = d

    rgb = frame_bgr[::stride, ::stride][sel][:, ::-1].astype(np.uint8)
    return pts, rgb


def write_ply(path: str | Path, pts: np.ndarray, rgb: np.ndarray) -> Path:
    path = Path(path)
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ]
    )
    records = np.zeros(n, dtype=dtype)
    records["x"], records["y"], records["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    records["red"], records["green"], records["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        records.tofile(fh)
    return path


def downsample_cloud(
    pts: np.ndarray,
    rgb: np.ndarray,
    voxel: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-downsample a cloud to keep the viewer light and snappy.

    `voxel` is the grid cell size in model units; `<=0` picks ~diag/1000
    (keeps point spacing proportional to object size, which the viewer
    renders at that spacing).
    """
    if len(pts) == 0:
        return pts, rgb
    if voxel <= 0.0:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        voxel = diag / 1000.0
        print(f"[voxel] auto size = {voxel:.3e}")
    pts, rgb = voxel_downsample(pts, rgb, voxel)
    print(f"[voxel] {len(pts):,} points")
    return pts, rgb


def _rotate_y(pts: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    return pts @ r.T


def capture_spin_sweep(
    predictor,
    *,
    source: int,
    n_frames: int,
    intrinsics: CameraIntrinsics,
    stride: int,
    keep: tuple[float, float],
    interval: float,
    preview: bool,
    voxel: float,
    out_path: str | Path,
) -> Path:
    """Capture a ~360-degree sweep under the turntable assumption.

    Best results with the subject centered and roughly mid-frame while the
    camera rotates around it. Rotating each frame's cloud about the camera
    Y-axis by its azimuth approximates a turntable shot.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {source}")

    parts_pts: list[np.ndarray] = []
    parts_rgb: list[np.ndarray] = []
    captured = 0
    try:
        print(f"[sweep] rotate the camera ~360deg around the subject; capturing {n_frames} frames")
        while captured < n_frames:
            ret, frame = cap.read()
            if not ret:
                break

            depth = predictor.infer(frame)
            pts, rgb = unproject(frame, depth, intrinsics, stride=stride, keep=keep)
            angle = 2.0 * np.pi * captured / n_frames
            parts_pts.append(_rotate_y(pts, -angle))
            parts_rgb.append(rgb)

            if preview:
                norm = normalize_depth(depth)
                view = np.hstack([frame, depth_to_bgr(norm)])
                cv2.putText(
                    view,
                    f"sweep {captured + 1}/{n_frames} - keep rotating, q to abort",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("sweep", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            captured += 1
            if interval > 0 and captured < n_frames:
                time.sleep(interval)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not parts_pts:
        raise RuntimeError("No frames captured for the sweep")

    pts = np.concatenate(parts_pts)
    rgb = np.concatenate(parts_rgb)
    print(f"[sweep] fused {captured} frames, {len(pts):,} points")
    pts, rgb = downsample_cloud(pts, rgb, voxel)
    return write_ply(out_path, pts, rgb)


def run_interactive_single(
    predictor,
    *,
    source: int,
    intrinsics: CameraIntrinsics,
    stride: int,
    keep: tuple[float, float],
    voxel: float,
    out_path: str | Path,
) -> Path:
    """Live preview; press 'c' to capture the current frame as a scan."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {source}")

    last_saved = ""
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            depth = predictor.infer(frame)
            norm = normalize_depth(depth)
            view = np.hstack([frame, depth_to_bgr(norm)])
            cv2.putText(
                view,
                f"c: capture  q: quit  |  {last_saved}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("scan", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("c"), ord("s")):
                pts, rgb = unproject(frame, depth, intrinsics, stride=stride, keep=keep)
                pts, rgb = downsample_cloud(pts, rgb, voxel)
                last_saved = str(write_ply(out_path, pts, rgb))
                print(f"[saved] {last_saved} ({len(pts):,} points)")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not last_saved:
        raise RuntimeError("No frame captured (press 'c' in the preview to scan)")
    return Path(last_saved)


def scan_image(
    predictor,
    image_path: str,
    *,
    intrinsics: CameraIntrinsics,
    stride: int,
    keep: tuple[float, float],
    voxel: float,
    out_path: str | Path,
) -> Path:
    """One-shot scan of a single image."""
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image {image_path}")
    depth = predictor.infer(frame)
    pts, rgb = unproject(frame, depth, intrinsics, stride=stride, keep=keep)
    print(f"[scan] {len(pts):,} points")
    pts, rgb = downsample_cloud(pts, rgb, voxel)
    return write_ply(out_path, pts, rgb)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scan-depth",
        description="Emit a colored 3D point cloud (.ply) from mono depth; view it with .venv/bin/python -m http.server 8000.",
    )
    p.add_argument("-o", "--out", default="scan.ply", help="output .ply path (default scan.ply)")
    p.add_argument("--image", metavar="PATH", help="scan a single image instead of the webcam")
    p.add_argument(
        "--capture",
        nargs="?",
        const="_auto_",
        default=None,
        metavar="DIR",
        help="interactively capture a scan session (default snapshots/session_<ts>)",
    )
    p.add_argument(
        "--sfm",
        metavar="DIR",
        help="free-motion scan: run COLMAP pose estimation + depth fusion on a captured session dir",
    )
    p.add_argument("--colmap-bin", metavar="PATH", default="colmap", help="path to the COLMAP binary")
    p.add_argument(
        "--sfm-backend",
        choices=["auto", "pycolmap", "colmap"],
        default="auto",
        help="pose estimator backend: auto = pycolmap wheel if installed else colmap CLI (default auto)",
    )
    p.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="voxel downsample size in model units, all capture modes (<=0: auto, ~diag/1000, default 0)",
    )
    p.add_argument("--sweep", type=int, default=0, metavar="N", help="capture N frames in a ~360deg sweep")
    p.add_argument("--source", type=int, default=0, help="webcam device index (default 0)")
    p.add_argument(
        "--probe-cameras",
        action="store_true",
        help="list available camera devices (indices usable with --source), then exit",
    )
    p.add_argument("--encoder", choices=["vits", "vitb", "vitl"], default="vitb", help="model size")
    p.add_argument("--input-size", type=int, default=518, help="model input resolution")
    p.add_argument("--fov", type=float, default=60.0, help="assumed horizontal camera FOV in degrees")
    p.add_argument("--stride", type=int, default=2, help="sampling stride (1 = all pixels, default 2)")
    p.add_argument(
        "--keep",
        nargs=2,
        type=float,
        default=(0.01, 0.99),
        metavar=("LO", "HI"),
        help="keep depth within these per-frame quantiles (default 0.01 0.99)",
    )
    p.add_argument(
        "--bg-ratio",
        type=float,
        default=2.0,
        metavar="R",
        help="drop pixels farther than R x the median SfM-tracked depth "
        "(0 disables; default 2.0)",
    )
    p.add_argument("--interval", type=float, default=0.6, help="seconds between sweep frames")
    p.add_argument("--no-preview", action="store_true", help="disable the live preview window")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.probe_cameras:
            devices = probe_cameras()
            if not devices:
                print("[probe] no camera devices found")
                return 1
            print(f"[probe] {len(devices)} camera device(s):")
            for d in devices:
                tag = ""
                if not d["frame"]:
                    tag = " (opened, frame grab FAILED)"
                print(
                    f"  --source {d['index']}: {d['name']} "
                    f"@ {d['width']}x{d['height']} {d['fps']:.0f}fps{tag}"
                )
            print("[probe] point --source at the index of your wanted camera, e.g. --source 1")
            failed = [d for d in devices if not d["frame"]]
            if failed:
                print(
                    "[probe] some devices enumerate but fail to grab frames — macOS "
                    "Continuity Camera support needs a newer OpenCV/AVFoundation build. "
                    "Workaround: shoot stills in the iPhone Camera app, AirDrop them into "
                    "the session's images/ folder, then run --sfm on it."
                )
            return 0

        predictor = DepthPredictor(encoder=args.encoder, input_size=args.input_size)
        print(f"[setup] device={predictor.device} encoder={args.encoder} input={predictor.input_size}")

        if args.sfm:
            session = Path(args.sfm)
            if not session.is_dir():
                raise ValueError(f"Session dir not found: {session}")
            images_dir = session / "images"
            if not images_dir.is_dir():
                images_dir = session
            print(f"[sfm] images: {images_dir}")
            sparse_dir = session / "colmap" / "sparse" / "0"
            if not (sparse_dir / "images.txt").is_file():
                sparse_dir = run_colmap(
                    images_dir,
                    session / "colmap",
                    binary=args.colmap_bin,
                    backend=args.sfm_backend,
                )
            else:
                print(f"[sfm] reusing existing sparse model: {sparse_dir}")
            pts, rgb = fuse_session(
                images_dir,
                sparse_dir,
                predictor,
                stride=args.stride,
                keep=tuple(args.keep),
                voxel=args.voxel,
                bg_ratio=args.bg_ratio,
            )
            write_ply(args.out, pts, rgb)
        elif args.capture is not None:
            out_dir = Path(args.capture) if args.capture != "_auto_" else None
            capture_session(predictor, source=args.source, out_dir=out_dir)
            print("[done] move to another angle, capture more frames, then run --sfm on the session dir")
            return 0
        elif args.image:
            w = cv2.imread(args.image)
            if w is None:
                raise RuntimeError(f"Could not read image {args.image}")
            h, w = w.shape[:2]
            intrinsics = CameraIntrinsics.from_fov(w, h, args.fov)
            scan_image(
                predictor,
                args.image,
                intrinsics=intrinsics,
                stride=args.stride,
                keep=tuple(args.keep),
                voxel=args.voxel,
                out_path=args.out,
            )
        elif args.sweep > 0:
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open capture source {args.source}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            cap.release()
            intrinsics = CameraIntrinsics.from_fov(w, h, args.fov)
            capture_spin_sweep(
                predictor,
                source=args.source,
                n_frames=args.sweep,
                intrinsics=intrinsics,
                stride=args.stride,
                keep=tuple(args.keep),
                interval=args.interval,
                preview=not args.no_preview,
                voxel=args.voxel,
                out_path=args.out,
            )
        else:
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open capture source {args.source}")
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            cap.release()
            intrinsics = CameraIntrinsics.from_fov(w, h, args.fov)
            run_interactive_single(
                predictor,
                source=args.source,
                intrinsics=intrinsics,
                stride=args.stride,
                keep=tuple(args.keep),
                voxel=args.voxel,
                out_path=args.out,
            )
        print(f"[done] view it: .venv/bin/python -m http.server 8000  # then open http://localhost:8000/viewer/ and load {args.out}")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
