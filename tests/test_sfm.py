import numpy as np
import pytest

from webcam_depth.sfm import (
    ColmapCamera,
    _quat_to_rot,
    estimate_scale,
    fuse_session,
    parse_reconstruction,
    run_colmap,
    voxel_downsample,
)


def write_minimal_reconstruction(tmp_path):
    """Images: view 1 = identity pose, view 2 = 90deg about Z. Both see point 1."""
    cam = ColmapCamera(1, "PINHOLE", 640, 480, np.array([500.0, 500.0, 320.0, 240.0]))
    point = np.array([0.5, 0.3, 2.0])
    gx, gy, gz = 0.0, 0.0, 0.0

    R2 = _quat_to_rot(0.7071067, 0.0, 0.0, 0.7071067)  # +90deg about +Z
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 240.0
    t2 = np.array([0.0, 0.0, 0.0])

    def project(p_world, rot3):
        p_cam = rot3 @ p_world
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        return u, v

    u1, v1 = project(point, np.eye(3))
    u2, v2 = project(point, R2)

    (tmp_path / "cameras.txt").write_text(
        "# Camera list\n1 PINHOLE 640 480 500.0 500.0 320.0 240.0\n"
    )
    (tmp_path / "points3D.txt").write_text(
        "1 0.5 0.3 2.0 255 255 255 0.0001 1 0 2 0\n"
    )
    (tmp_path / "images.txt").write_text(
        "# Image list\n"
        f"1 {1.0:.15g} {gx:.15g} {gy:.15g} {gz:.15g} 0.0 0.0 0.0 1 frame_0001.png\n"
        f"{u1:.3f} {v1:.3f} 1\n"
        f"2 0.7071067 0.0 0.0 0.7071067 {t2[0]} {t2[1]} {t2[2]} 1 frame_0002.png\n"
        f"{u2:.3f} {v2:.3f} 1\n"
    )
    return cam, point


def test_quat_rotation_roundtrip():
    # rotate +90deg about Z maps (1,0,0) -> (0,1,0)
    R = _quat_to_rot(0.7071067, 0.0, 0.0, 0.7071067)
    out = R @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(out, [0.0, 1.0, 0.0], atol=1e-4)


def test_parse_reconstruction(tmp_path):
    cam, _ = write_minimal_reconstruction(tmp_path)
    rec = parse_reconstruction(tmp_path)

    assert rec.cameras[1].model == "PINHOLE"
    assert np.allclose(rec.cameras[1].params, cam.params)
    assert rec.points3d[1].xyz is not None
    assert len(rec.images) == 2

    img1 = next(i for i in rec.images if i.name == "frame_0001.png")
    assert np.allclose(img1.rotation_matrix(), np.eye(3), atol=1e-4)
    assert np.allclose(img1.camera_center(), [0, 0, 0], atol=1e-4)


def test_intrinsics_from_camera():
    cam = ColmapCamera(1, "PINHOLE", 640, 480, np.array([500.0, 510.0, 320.0, 240.0]))
    intrin = cam.intrinsics()
    assert intrin.fx == 500.0 and intrin.fy == 510.0
    assert intrin.cx == 320.0 and intrin.cy == 240.0


def test_reprojection_matches_observations(tmp_path):
    write_minimal_reconstruction(tmp_path)
    rec = parse_reconstruction(tmp_path)
    cam = rec.cameras[1]
    fx, fy, cx, cy = (float(v) for v in cam.params)
    for img in rec.images:
        R = img.rotation_matrix()
        pt = rec.points3d[1]
        p_cam = R @ pt.xyz + img.tvec
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        assert abs(u - img.x2d[0]) < 1e-2
        assert abs(v - img.y2d[0]) < 1e-2


def test_estimate_scale_recovers_known_scale():
    # true camera depth of the point is 2.0; depth map stores relative depth 0.5
    depth = np.zeros((480, 640), dtype=np.float32)
    samples = [(440, 315, 2.0)]  # z_cam from sfm
    depth[315, 440] = 0.5
    scale = estimate_scale(depth, samples, min_tracks=1)
    assert scale is not None
    assert abs(scale - 0.25) < 1e-6


def test_estimate_scale_returns_none_when_no_tracks():
    depth = np.zeros((10, 10), dtype=np.float32)
    assert estimate_scale(depth, [], min_tracks=1) is None


def test_voxel_downsample_averages():
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    out_pts, out_rgb = voxel_downsample(pts, rgb, voxel=2.0)
    assert len(out_pts) == 1
    assert np.allclose(out_pts[0], [0.5, 0.5, 0.5])
    assert np.allclose(out_rgb[0], [25, 35, 45])


def test_voxel_downsample_separates_cells():
    pts = np.array([[0, 0, 0], [5, 5, 5]], dtype=np.float32)
    rgb = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.uint8)
    out_pts, _ = voxel_downsample(pts, rgb, voxel=2.0)
    assert len(out_pts) == 2


def test_voxel_zero_noop():
    pts = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    rgb = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.uint8)
    o, _ = voxel_downsample(pts, rgb, voxel=0.0)
    assert len(o) == 2


def test_run_colmap_missing_binary_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="brew install colmap"):
        run_colmap(tmp_path, tmp_path / "work", backend="colmap")


def test_run_colmap_backend_auto_prefers_pycolmap(tmp_path, monkeypatch):
    # with pycolmap importable and no binary present, backend=auto must not
    # fall through to the binary path
    monkeypatch.setattr("shutil.which", lambda _: None)
    from webcam_depth import sfm

    try:
        import pycolmap  # noqa: F401
    except ImportError:
        pytest.skip("pycolmap not installed")
    monkeypatch.setattr(sfm, "_pycolmap_available", lambda: True)
    monkeypatch.setattr(sfm, "_run_colmap_pycolmap", lambda *a, **k: "sparse")

    assert run_colmap(tmp_path, tmp_path / "work") == "sparse"


def test_run_colmap_rejects_bad_matcher(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/colmap")
    with pytest.raises(ValueError, match="matcher"):
        run_colmap(tmp_path, tmp_path / "work", matcher="nope")


def test_run_colmap_invokes_steps(tmp_path, monkeypatch):
    import types

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("webcam_depth.sfm.subprocess.run", fake_run)
    monkeypatch.setattr("webcam_depth.sfm.shutil.which", lambda _: "/usr/bin/colmap")
    images = tmp_path / "images"
    images.mkdir()
    model = tmp_path / "work" / "sparse" / "0"
    model.mkdir(parents=True)

    out = run_colmap(images, tmp_path / "work", backend="colmap")
    assert out == model
    assert len(calls) == 3
    tokens = [t for cmd in calls for t in cmd]
    assert tokens.count("feature_extractor") == 1
    assert tokens.count("exhaustive_matcher") == 1
    assert tokens.count("mapper") == 1
    assert "--ImageReader.camera_model" in tokens and "PINHOLE" in tokens

def test_fuse_session_reconstructs_world_planes(tmp_path):
    """Two views of a 3D scene (base plane z=3 + raised platform z=2.6) with
    per-frame depth drift; fusion must place every point back on its plane."""
    import cv2

    f, cx, cy, W, H = 200.0, 128.0, 128.0, 256, 256
    BASE_Z, PLAT_Z = 3.0, 2.6
    PLAT = (-0.5, 0.6, -0.2, 0.5)  # platform region x0 x1 y0 y1

    def rot_y(deg):
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    R1 = np.eye(3)
    R2 = rot_y(30.0)
    t1 = np.zeros(3)
    t2 = np.array([0.10, -0.05, 0.20])

    def section(x, y):
        return np.where(
            (PLAT[0] <= x) & (x <= PLAT[1]) & (PLAT[2] <= y) & (y <= PLAT[3]),
            PLAT_Z,
            BASE_Z,
        )

    guides = [
        (-0.6, 0.4),
        (0.6, 0.35),
        (-0.45, -0.3),
        (0.5, -0.4),
        (0.0, 0.0),
        (-0.3, 0.2),
        (0.3, -0.15),
        (0.1, 0.3),
        (-0.1, -0.45),
    ]
    pts3d = {i + 1: np.array([gx, gy, section(gx, gy)], dtype=np.float64) for i, (gx, gy) in enumerate(guides)}

    def obs(world, r, tv):
        p_cam = r @ world + tv
        return (f * p_cam[0] / p_cam[2] + cx, f * p_cam[1] / p_cam[2] + cy)

    images_dir = tmp_path / "images"
    sparse_dir = tmp_path / "sparse"
    images_dir.mkdir()
    sparse_dir.mkdir()

    line1 = ["1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 frame_0001.png"]
    line2 = ["2 0.9659258 0.0 0.2588190 0.0 0.1 -0.05 0.2 1 frame_0002.png"]
    track1, track2 = [], []
    for pid, w in pts3d.items():
        u1, v1 = obs(w, R1, t1)
        u2, v2 = obs(w, R2, t2)
        track1.append(f"{u1:.6f} {v1:.6f} {pid}")
        track2.append(f"{u2:.6f} {v2:.6f} {pid}")
    line1.append(" ".join(track1))
    line2.append(" ".join(track2))

    (sparse_dir / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {f} {f} {cx} {cy}\n")
    (sparse_dir / "points3D.txt").write_text(
        "\n".join(
            f"{pid} {w[0]} {w[1]} {w[2]} 200 200 200 0.0001 1 0 2 0"
            for pid, w in pts3d.items()
        )
        + "\n"
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(line1) + "\n" + "\n".join(line2) + "\n"
    )

    one = np.tile(np.array([30, 60, 10], dtype=np.uint8), (H, W, 1))
    two = np.tile(np.array([90, 60, 90], dtype=np.uint8), (H, W, 1))
    cv2.imwrite(str(images_dir / "frame_0001.png"), one)
    cv2.imwrite(str(images_dir / "frame_0002.png"), two)

    scales = {0: 0.6, 1: 0.85}

    class FakePredictor:
        def infer(self, frame):
            idx = 0 if frame[0, 0, 2] == 10 else 1
            R = R1 if idx == 0 else R2
            tv = t1 if idx == 0 else t2
            center = -(R.T @ tv)
            us, vs = np.meshgrid(np.arange(W), np.arange(H))
            dir_cam = np.stack(
                [(us - cx) / f, (vs - cy) / f, np.ones_like(us)], axis=-1
            )
            dir_world = dir_cam @ R
            lam = np.zeros_like(us, dtype=np.float64)
            for z0 in (BASE_Z, PLAT_Z):
                l0 = (z0 - center[2]) / dir_world[..., 2]
                hit = center[None, None, :] + l0[..., None] * dir_world
                lam = np.maximum(lam, np.where(section(hit[..., 0], hit[..., 1]) == z0, l0, 0))
            return lam * (scales.get(idx, 1.0))

    pts, rgb = fuse_session(
        images_dir, sparse_dir, FakePredictor(), stride=1, keep=(0.0, 1.0), voxel=0.0
    )
    assert pts.shape[1] == 3
    assert 0.0 <= rgb.min() and rgb.max() <= 255
    for x, y in ((0.0, 0.0), (0.0, -0.5)):
        m = np.hypot(pts[:, 0] - x, pts[:, 1] - y) < 0.05
        assert m.any()
        target = section(x, y)
        assert abs(float(np.median(pts[m, 2])) - target) < 0.08, (
            f"fused z median {np.median(pts[m, 2]):.3f} not on plane z={target}"
        )
    assert pts[:, 2].min() < 2.7 and pts[:, 2].max() > 2.9  # both planes survived
