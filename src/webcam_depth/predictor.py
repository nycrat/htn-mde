from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_IDS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


def pick_device() -> torch.device:
    """Prefer the Mac GPU (Metal) when available, else fall back to CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DepthPredictor:
    """Depth Anything V2 wrapper: BGR frame in, full-resolution depth map out."""

    def __init__(
        self,
        encoder: str = "vitb",
        input_size: int = 518,
        device: torch.device | None = None,
    ) -> None:
        if encoder not in MODEL_IDS:
            raise ValueError(f"Unknown encoder {encoder!r}; choose from {sorted(MODEL_IDS)}")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        self.encoder = encoder
        self.device = device or pick_device()
        self.input_size = int(input_size)

        model_id = MODEL_IDS[encoder]
        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        if self.input_size != 518:
            self.image_processor.size = {"shortest_edge": self.input_size}
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(self.device).eval()

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Return a depth map (float32) at the same resolution as the input frame."""
        rgb = np.ascontiguousarray(frame_bgr[..., ::-1])
        inputs = self.image_processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.model(**inputs)
        depth = out.predicted_depth.squeeze(0)
        h, w = frame_bgr.shape[:2]
        depth = torch.nn.functional.interpolate(
            depth[None, None],
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return depth.detach().cpu().numpy().astype(np.float32)