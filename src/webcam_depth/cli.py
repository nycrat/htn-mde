from __future__ import annotations

import argparse
import sys

from .pipeline import run_single_image, run_webcam
from .predictor import DepthPredictor


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="webcam-depth",
        description="Real-time monocular depth estimation on Apple Silicon (Depth Anything V2).",
    )
    p.add_argument(
        "--encoder",
        choices=["vits", "vitb", "vitl"],
        default="vitb",
        help="model size: vits = fast, vitb = quality (default), vitl = max quality",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="shortest-edge input resolution for the model (higher = finer details, slower)",
    )
    p.add_argument(
        "--image",
        metavar="PATH",
        help="run on a single image and exit instead of the live camera loop",
    )
    p.add_argument(
        "--source",
        type=int,
        default=0,
        help="webcam device index (default 0)",
    )
    p.add_argument(
        "--smooth",
        type=float,
        default=0.35,
        help="temporal smoothing alpha in [0, 1]; 0 disables (default 0.35)",
    )
    p.add_argument(
        "--snapshots",
        metavar="DIR",
        default="snapshots",
        help="directory for 's'-key snapshots / single-image output (default snapshots)",
    )
    p.add_argument("--record", metavar="PATH", help="write the processed stream to an mp4")
    p.add_argument("--flip", action="store_true", help="mirror the camera input horizontally")
    p.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=[1280, 720],
        metavar=("W", "H"),
        help="requested capture resolution, e.g. --resolution 1920 1080 (default 1280 720)",
    )
    p.add_argument("--device", choices=["mps", "cpu"], default=None, help="force a device")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictor = DepthPredictor(
            encoder=args.encoder,
            input_size=args.input_size,
            device=args.device,
        )
        print(f"[setup] device={predictor.device} encoder={args.encoder} input={predictor.input_size}")
        if args.image:
            run_single_image(predictor, args.image, out_dir=args.snapshots)
        else:
            run_webcam(
                predictor,
                source=args.source,
                smooth_alpha=args.smooth,
                snap_dir=args.snapshots,
                record_path=args.record,
                flip=args.flip,
                width=args.resolution[0],
                height=args.resolution[1],
            )
    except (RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())