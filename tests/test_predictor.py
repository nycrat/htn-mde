import numpy as np

from webcam_depth.pipeline import DepthSmoother, depth_to_bgr, normalize_depth
from webcam_depth.predictor import MODEL_IDS


def test_model_ids_cover_cli_choices():
    assert set(MODEL_IDS) == {"vits", "vitb", "vitl"}


def test_normalize_depth_range():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.2, 8.0, size=(64, 64))
    norm = normalize_depth(depth)
    assert norm.shape == depth.shape
    assert norm.dtype == np.float32
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm.max() > 0.5


def test_normalize_depth_degenerate():
    flat = np.full((16, 16), 3.0, dtype=np.float32)
    norm = normalize_depth(flat)
    assert np.all(norm == 0.0)


def test_depth_to_bgr():
    norm = np.linspace(0.0, 1.0, 128 * 128).reshape(128, 128).astype(np.float32)
    bgr = depth_to_bgr(norm)
    assert bgr.shape == (128, 128, 3)
    assert bgr.dtype == np.uint8


def test_smoother_ema():
    s = DepthSmoother(alpha=0.4)
    first = np.ones((8, 8), dtype=np.float32)
    second = np.zeros((8, 8), dtype=np.float32)
    assert np.all(s.update(first) == 1.0)
    smoothed = s.update(second)
    assert smoothed.shape == (8, 8)
    assert 0.0 < smoothed[0, 0] < 1.0


def test_smoother_disabled():
    s = DepthSmoother(alpha=0.0)
    one = np.ones((8, 8), dtype=np.float32)
    zero = np.zeros((8, 8), dtype=np.float32)
    s.update(one)
    assert np.all(s.update(zero) == 0.0)