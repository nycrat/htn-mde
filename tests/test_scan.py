import struct

import numpy as np

from webcam_depth.scan import CameraIntrinsics, unproject, write_ply


def test_intrinsics_center_backprojects_to_origin():
    intrin = CameraIntrinsics.from_fov(640, 480, 60.0)
    depth = np.full((480, 640), 2.0, dtype=np.float32)
    pts, _ = unproject(
        np.zeros((480, 640, 3), dtype=np.uint8),
        depth,
        intrin,
        stride=1,
        keep=(0.0, 1.0),
    )
    cx, cy = 320, 240
    idx = cy * 640 + cx
    assert pts[idx, 0] == 0.0
    assert pts[idx, 1] == 0.0
    assert pts[idx, 2] == 2.0
    assert idx < len(pts)


def test_intrinsics_maps_focal_scale():
    intrin = CameraIntrinsics(fx=120.0, fy=120.0, cx=150.0, cy=50.0)
    depth = np.full((100, 300), 3.0, dtype=np.float32)
    pts, _ = unproject(
        np.zeros((100, 300, 3), dtype=np.uint8),
        depth,
        intrin,
        stride=1,
        keep=(0.0, 1.0),
    )
    idx = 50 * 300 + 270  # one focal length right of center (cx=150, fx=120)
    assert abs(pts[idx, 0] - 3.0) < 1e-3
    assert pts[idx, 1] == 0.0
    assert pts[idx, 2] == 3.0


def test_unproject_shapes_and_colors():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    depth = rng.uniform(1.0, 8.0, (60, 80)).astype(np.float32)
    intrin = CameraIntrinsics(fx=100.0, fy=100.0, cx=40.0, cy=30.0)
    pts, rgb = unproject(frame, depth, intrin, stride=2, keep=(0.0, 1.0))
    assert pts.shape[1] == 3
    assert pts.dtype == np.float32
    assert rgb.shape == (pts.shape[0], 3)
    assert rgb.dtype == np.uint8


def test_unproject_quantile_filter_drops_extremes():
    depth = np.full((40, 40), 5.0, dtype=np.float32)
    depth[0, 0] = 1000.0  # an outlier far point
    intrin = CameraIntrinsics(fx=100.0, fy=100.0, cx=20.0, cy=20.0)
    pts, _ = unproject(
        np.zeros((40, 40, 3), dtype=np.uint8),
        depth,
        intrin,
        stride=1,
        keep=(0.0, 0.99),
    )
    assert pts[:, 2].max() < 6.0


def test_write_ply_roundtrip(tmp_path):
    pts = np.array([[0, 0, 0], [1, 2, 3], [-4, 5, 6]], dtype=np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    out = write_ply(tmp_path / "t.ply", pts, rgb)
    with open(out, "rb") as fh:
        head = b""
        while b"end_header" not in head:
            head += fh.readline()
        assert head.startswith(b"ply")
        assert b"format binary_little_endian 1.0" in head
        assert b"element vertex 3" in head
        data = fh.read()
    n = len(data) // (3 * 4 + 3)
    assert n == 3
    view = np.frombuffer(data, dtype=np.uint8).reshape(n, -1)
    assert view[0][:4].tobytes() == struct.pack("<f", 0.0)
    assert view[0][12:15].tobytes() == b"\xff\x00\x00"
    assert view[2][12:15].tobytes() == b"\x00\x00\xff"