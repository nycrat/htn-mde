from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .pipeline import depth_to_bgr, normalize_depth
from .predictor import DepthPredictor

if TYPE_CHECKING:
    from .scan import CameraIntrinsics


COLMAP_INSTALL_HINT = (
    "COLMAP could not be found. Install it with `brew install colmap` "
    "and retry (or pass --colmap-bin /path/to/colmap)."
)


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def intrinsics(self) -> CameraIntrinsics:
        from .scan import CameraIntrinsics as CI

        fx, fy, cx, cy = (float(v) for v in self.params)
        return CI(fx, fy, cx, cy)


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # (4,) world->camera quaternion (qw, qx, qy, qz)
    tvec: np.ndarray  # (3,) world->camera translation
    camera_id: int
    name: str
    x2d: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    y2d: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    point3d_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def rotation_matrix(self) -> np.ndarray:
        return _quat_to_rot(*self.qvec)

    def camera_center(self) -> np.ndarray:
        return -(self.rotation_matrix().T @ self.tvec)


@dataclass
class ColmapPoint3D:
    id: int
    xyz: np.ndarray  # (3,)
    rgb: np.ndarray  # (3,) uint8
    error: float


@dataclass
class ColmapReconstruction:
    cameras: dict[int, ColmapCamera] = field(default_factory=dict)
    images: list[ColmapImage] = field(default_factory=list)
    points3d: dict[int, ColmapPoint3D] = field(default_factory=dict)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    w, x, y, z = float(qw), float(qx), float(qy), float(qz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def parse_cameras(path: str | Path) -> dict[int, ColmapCamera]:
    out: dict[int, ColmapCamera] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            cam = ColmapCamera(
                id=int(toks[0]),
                model=toks[1],
                width=int(toks[2]),
                height=int(toks[3]),
                params=np.asarray(toks[4:], dtype=np.float32),
            )
            out[cam.id] = cam
    return out


def parse_images(path: str | Path) -> list[ColmapImage]:
    images: list[ColmapImage] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if len(lines) % 2 != 0:
        raise ValueError("Malformed images.txt: expected alternating metadata/track lines")
    for meta, track in zip(lines[::2], lines[1::2]):
        mt = meta.split()
        img = ColmapImage(
            id=int(mt[0]),
            qvec=np.asarray(mt[1:5], dtype=np.float64),
            tvec=np.asarray(mt[5:8], dtype=np.float64),
            camera_id=int(mt[8]),
            name=mt[9],
        )
        tt = track.split()
        img.x2d = np.asarray(tt[0::3], dtype=np.float32)
        img.y2d = np.asarray(tt[1::3], dtype=np.float32)
        ids = np.asarray(tt[2::3], dtype=np.int64)
        img.point3d_ids = ids
        images.append(img)
    return images


def parse_points3d(path: str | Path) -> dict[int, ColmapPoint3D]:
    out: dict[int, ColmapPoint3D] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            out[int(toks[0])] = ColmapPoint3D(
                id=int(toks[0]),
                xyz=np.asarray(toks[1:4], dtype=np.float64),
                rgb=np.asarray(toks[4:7], dtype=np.uint8),
                error=float(toks[7]),
            )
    return out


def parse_reconstruction(sparse_dir: str | Path) -> ColmapReconstruction:
    d = Path(sparse_dir)
    cameras = parse_cameras(d / "cameras.txt")
    images = parse_images(d / "images.txt")
    points3d = parse_points3d(d / "points3D.txt")
    if not cameras:
        raise ValueError(f"No cameras parsed from {d}")
    return ColmapReconstruction(cameras=cameras, images=images, points3d=points3d)


def _run_colmap_pycolmap(
    images_dir: str | Path,
    work_dir: str | Path,
    *,
    matcher: str = "exhaustive",
) -> Path:
    """In-process COLMAP via the pip-installable pycolmap binding."""
    # pycolmap then torch in one process collides on libomp unless relaxed.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import pycolmap
    except ImportError:
        raise RuntimeError(
            "backend='pycolmap' requested but pycolmap is not installed — "
            "run `pip install pycolmap` (or use backend='colmap' and the homebrew binary)"
        )

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    db = wd / "database.db"
    if db.exists():
        db.unlink()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "PINHOLE"
    reader.camera_params = ""

    print("[colmap] extract_features (pycolmap)...")
    pycolmap.extract_features(
        str(db),
        str(Path(images_dir)),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
    )
    matcher_fn = pycolmap.match_exhaustive if matcher == "exhaustive" else pycolmap.match_sequential
    print(f"[colmap] {matcher} (pycolmap)...")
    matcher_fn(str(db))

    print("[colmap] incremental_mapping (pycolmap)...")
    sparse_out = wd / "sparse"
    maps = pycolmap.incremental_mapping(str(db), str(Path(images_dir)), str(sparse_out))
    if not maps:
        raise RuntimeError("pycolmap produced no model — the object likely lacks texture or overlap")

    best_idx = max(maps, key=lambda idx: maps[idx].num_reg_images())
    best = maps[best_idx]
    model_dir = sparse_out / "0"
    model_dir.mkdir(parents=True, exist_ok=True)
    best.write_text(str(model_dir))
    print(f"[colmap] best model: {best.num_reg_images()} registered images, {best.num_points3D()} points")
    return model_dir


def _pycolmap_available() -> bool:
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        return False
    return True


def run_colmap(
    images_dir: str | Path,
    work_dir: str | Path,
    *,
    binary: str = "colmap",
    matcher: str = "exhaustive",
    camera_model: str = "PINHOLE",
    max_image_size: int = 2000,
    backend: str = "auto",
) -> Path:
    """Run COLMAP feature_extractor + matcher + mapper on a set of images.

    Returns the path of the sparse model directory (contains cameras.txt,
    images.txt, points3D.txt). Images are fed raw with a single-shared
    PINHOLE camera model; refinement recovers focal length on its own.

    ``backend``: "auto" prefers the ``pycolmap`` wheel (no system install),
    falling back to the ``colmap`` CLI binary; "pycolmap" and "colmap" force
    one or the other.
    """
    if matcher not in ("exhaustive", "sequential"):
        raise ValueError(f"Unknown matcher {matcher!r}; choose from exhaustive, sequential")

    if backend in ("auto", "pycolmap"):
        if _pycolmap_available():
            return _run_colmap_pycolmap(images_dir, work_dir, matcher=matcher)
        if backend == "pycolmap":
            raise RuntimeError("pycolmap is not installed — run `pip install pycolmap`")

    colmap = shutil.which(binary)
    if colmap is None:
        raise RuntimeError(COLMAP_INSTALL_HINT)

    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    db = wd / "database.db"
    if db.exists():
        db.unlink()

    extract = [
        colmap, "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(Path(images_dir)),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_image_size", str(max_image_size),
    ]
    match_cmd = [
        colmap, "exhaustive_matcher" if matcher == "exhaustive" else "sequential_matcher",
        "--database_path", str(db),
    ]
    sparse_out = wd / "sparse"
    map_cmd = [
        colmap, "mapper",
        "--database_path", str(db),
        "--image_path", str(Path(images_dir)),
        "--output_path", str(sparse_out),
    ]

    for step, cmd in (("feature_extractor", extract), (matcher, match_cmd), ("mapper", map_cmd)):
        print(f"[colmap] {step} ...")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(
                f"COLMAP {step} failed (exit {proc.returncode}). "
                f"Common fixes: more overlap/texture between frames, or fewer images.\n"
                f"stderr tail:\n{proc.stderr.strip().splitlines()[-8:]}"
            )

    model_dir = sparse_out / "0"
    if not model_dir.is_dir():
        raise RuntimeError("COLMAP produced no sparse model — the object likely lacks texture or overlap")
    return model_dir


def _reproject_tracks(
    image: ColmapImage,
    camera: ColmapCamera,
    points3d: dict[int, ColmapPoint3D],
    *,
    max_reproj_px: float = 3.0,
) -> list[tuple[int, int, float, int]]:
    """Recover (px, py, z_cam, point3d_id) samples for a registered image.

    Keeps observations that reproject into the image with low error, so scale
    estimation only sees trustworthy, in-bounds geometry.
    """
    fx, fy, cx, cy = (float(v) for v in camera.params)
    R = image.rotation_matrix()
    samples: list[tuple[int, int, float, int]] = []
    for u2d, v2d, pid in zip(image.x2d, image.y2d, image.point3d_ids):
        if pid < 0:
            continue
        pt = points3d.get(int(pid))
        if pt is None:
            continue
        p_cam = R @ pt.xyz + image.tvec
        if p_cam[2] <= 0.0:
            continue
        proj = np.array([fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy])
        if np.hypot(proj[0] - u2d, proj[1] - v2d) > max_reproj_px:
            continue
        if not (0 <= round(u2d) < camera.width and 0 <= round(v2d) < camera.height):
            continue
        samples.append((round(u2d), round(v2d), float(p_cam[2]), int(pid)))
    return samples


def estimate_scale(
    depth: np.ndarray,
    samples: list[tuple[int, int, float]],
    *,
    min_tracks: int = 3,
) -> float | None:
    """Per-frame relative scale via s_i = median(depth[pixel] / z_sfm).

    Depth Anything maps are relative (scale varies between frames); the SfM
    sparse points pin that scale down for this view. Returns None when there
    are too few usable tracks.
    """
    h, w = depth.shape[:2]
    ratios: list[float] = []
    for px, py, z_cam in samples:
        if not (0 <= px < w and 0 <= py < h):
            continue
        d = float(depth[py, px])
        if np.isfinite(d) and d > 0.0 and z_cam > 0.0:
            ratios.append(d / z_cam)
    if len(ratios) < min_tracks:
        return None
    return float(np.median(ratios))


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares 3D similarity dst = scale*R @ src + t (Umeyama)."""
    if len(src) < 3:
        raise ValueError("similarity fit needs at least 3 correspondences")
    mx, my = src.mean(axis=0), dst.mean(axis=0)
    Xc, Yc = src - mx, dst - my
    cov = Yc.T @ Xc  # row-vector convention: dst = scale*(src@R.T) + t
    U, _, Vh = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(U @ Vh))
    R = U @ np.diag([1.0, 1.0, d]) @ Vh
    Yr = Xc @ R.T
    scale = float(np.sum(Yc * Yr) / max(np.sum(Yr * Yr), 1e-12))
    t = my - scale * (R @ mx)
    return scale, R, t


def fuse_session(
    images_dir: str | Path,
    sparse_dir: str | Path,
    predictor: DepthPredictor,
    *,
    stride: int = 2,
    keep: tuple[float, float] = (0.01, 0.99),
    voxel: float = 0.0,
    bg_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Align each frame's Depth-Anything cloud into one shared SfM world.

    SfM only recovers camera poses up to a global similarity, so for every
    registered frame we fit the 3D similarity (scale + rigid motion) that maps
    its sparse SfM points onto the same points unprojected from the depth map.
    Applying the inverse transform to the whole frame cloud merges all frames
    in the (arbitrary, SfM-normalized) world frame.

    Depth-Anything's depth is relative with an unbounded far end, so distant
    background and stray grazing-ray spikes invert to wildly oversized world
    coordinates. `bg_ratio` keeps only pixels within `bg_ratio` x the median
    SfM-tracked depth of the frame, culling that overflow (0 disables).

    Returns (points, rgb) across all registered frames, voxel-downsampled.
    Reads a saved `depth/<name>.npy` next to `images_dir` when present
    (written by `capture_session`) instead of re-running inference.
    """
    rec = parse_reconstruction(sparse_dir)
    if not rec.images:
        raise RuntimeError("No registered images in the sparse model")

    images_dir = Path(images_dir)
    depth_dir = images_dir.parent / "depth"

    from .scan import unproject

    all_pts: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []
    skipped: list[str] = []

    for image in rec.images:
        camera = rec.cameras.get(image.camera_id)
        if camera is None:
            skipped.append(image.name)
            continue
        frame_path = images_dir / image.name
        frame = cv2.imread(str(frame_path))
        if frame is None:
            skipped.append(image.name)
            continue

        depth_path = depth_dir / f"{Path(image.name).stem}.npy"
        if depth_path.is_file():
            depth = np.load(depth_path)
        else:
            depth = predictor.infer(frame)

        samples = _reproject_tracks(image, camera, rec.points3d, max_reproj_px=3.0)
        if len(samples) < 3:
            skipped.append(image.name)
            continue

        h, w = depth.shape[:2] if hasattr(depth, "shape") else (0, 0)
        instr = camera.intrinsics()
        src, dst = [], []
        for px, py, _z, pid in samples:
            if 0 <= px < w and 0 <= py < h:
                d = float(depth[py, px])
                if np.isfinite(d) and d > 0.0:
                    src.append(rec.points3d[pid].xyz)
                    dst.append([(px - instr.cx) * d / instr.fx, (py - instr.cy) * d / instr.fy, d])
        if len(dst) < 3:
            skipped.append(image.name)
            continue
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        max_track_depth = float(np.median(dst[:, 2]))  # robust object-depth reference

        # Hard anchor: the SfM camera center coincides with the depth-frame
        # origin. Replicating it keeps near-coplanar track sets from sloshing
        # the out-of-plane rotation/scale.
        center = image.camera_center()
        anchor = max(3, min(len(dst) * 3, 60))
        src = np.vstack([src, np.tile(center, (anchor, 1))])
        dst = np.vstack([dst, np.zeros((anchor, 3))])

        scale, R, t = _fit_similarity(src, dst)
        fit_resid = float(np.median(np.linalg.norm((scale * (src @ R.T) + t) - dst, axis=1)))

        pts, rgb = unproject(frame, depth, instr, stride=stride, keep=keep)
        if bg_ratio > 0.0:
            keep_far = pts[:, 2] <= bg_ratio * max_track_depth
            pts, rgb = pts[keep_far], rgb[keep_far]
        pts = (pts - t) @ R / scale  # inverse of dst = scale*R@src + t
        all_pts.append(pts)
        all_rgb.append(rgb)
        print(f"[fusion] {image.name}: {len(dst):,} tracks, fit_resid={fit_resid:.3f} -> {len(pts):,} pts")

    if skipped:
        print(f"[fusion] skipped frames without pose alignment: {len(skipped)} ({', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''})")
    if not all_pts:
        raise RuntimeError("No frame could be aligned/fused — need more texture or overlap between frames")

    pts = np.concatenate(all_pts)
    rgb = np.concatenate(all_rgb)
    if voxel <= 0.0:
        diag = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        voxel = diag / 1000.0
        print(f"[fusion] auto voxel size = {voxel:.3e}")
    pts, rgb = voxel_downsample(pts, rgb, voxel)
    print(f"[fusion] fused {len(all_pts)} frames -> {len(pts):,} points")
    return pts, rgb


def voxel_downsample(
    pts: np.ndarray,
    rgb: np.ndarray,
    voxel: float,
    *,
    color_dtype=np.uint8,
) -> tuple[np.ndarray, np.ndarray]:
    """Average points + colors inside `voxel`-sized grid cells (numpy only)."""
    if voxel <= 0.0 or len(pts) == 0:
        return pts, rgb
    mins = pts.min(axis=0)
    key = np.floor((pts - mins) / voxel).astype(np.int64)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    summed = np.zeros((len(counts), 3), dtype=np.float64)
    colors = np.zeros((len(counts), 3), dtype=np.float64)
    np.add.at(summed, inv, pts.astype(np.float64))
    np.add.at(colors, inv, rgb.astype(np.float64))
    out = (summed / counts[:, None]).astype(np.float32)
    col = (colors / counts[:, None]).round().astype(color_dtype)
    return out, col


def capture_session(
    predictor: DepthPredictor,
    *,
    source: int = 0,
    out_dir: str | Path | None = None,
    overlap_warn: float = 0.22,
) -> Path:
    """Live preview + interactive scan-session capture.

    Press `c` to append the current frame (and its depth) to the session;
    press `q` to finish. Frames land in `<out>/images/`, depth as .npy in
    `<out>/depth/`, ready for `scan-depth --sfm <out>`.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open capture source {source}")

    if out_dir is None:
        out_dir = Path("snapshots") / f"session_{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    dep_dir = out_dir / "depth"
    img_dir.mkdir(parents=True, exist_ok=True)
    dep_dir.mkdir(parents=True, exist_ok=True)

    last_gray: np.ndarray | None = None
    captured = 0
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
                f"c: capture  q: quit  |  {captured} frames",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("capture", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("c"), ord("s")):
                if last_gray is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (160, 90))
                    diff = float(np.mean(np.abs(gray.astype(np.float32) - last_gray)) / 255.0)
                    if diff > overlap_warn:
                        print(f"[capture] notice: view changed a lot (Δ={diff:.2f}); move slowly and keep ~70% overlap")
                last_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))

                stem = f"frame_{captured:04d}"
                cv2.imwrite(str(img_dir / f"{stem}.png"), frame)
                np.save(dep_dir / f"{stem}.npy", depth)
                captured += 1
                print(f"[capture] {captured}: {img_dir / stem}.png")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured == 0:
        raise RuntimeError("No frames captured (press 'c' to add frames to the session)")
    print(f"[capture] session ready: {out_dir} ({captured} frames)")
    return out_dir