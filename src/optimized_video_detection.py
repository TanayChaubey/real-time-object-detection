"""Lower-latency YOLO video inference with configurable input resolution.

This keeps one model instance loaded, uses a smaller inference size by
default, optionally enables FP16 on CUDA, and reports measured throughput.
The original video_detection.py remains the baseline implementation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from src.preprocessing import validate_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimized YOLO video inference.")
    parser.add_argument("--source", required=True, help="Video path or camera index.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA only.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Process every Nth frame; skipped frames are not inferred.")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", default="outputs")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1.")
    if args.imgsz <= 0 or args.frame_stride < 1 or args.max_det < 1:
        raise ValueError("imgsz, frame-stride and max-det must be positive.")

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    use_half = bool(args.half and torch.cuda.is_available() and device != "cpu")
    model = YOLO(args.model)  # Load once; do not reload per frame.

    writer = None
    output_path = None
    if args.save:
        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = "webcam" if isinstance(source, int) else Path(str(source)).stem
        output_path = out_dir / f"{stem}_optimized.mp4"

    frame_index = 0
    processed = 0
    inference_seconds = 0.0
    total_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if (frame_index - 1) % args.frame_stride:
                continue

            frame = validate_frame(frame)
            start = time.perf_counter()
            result = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                half=use_half,
                max_det=args.max_det,
                verbose=False,
            )[0]
            if str(device).startswith("0") and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            processed += 1
            annotated = result.plot()

            if not args.no_display:
                cv2.imshow("Optimized Object Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not 0 < fps <= 120:
                        fps = 25.0
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (w, h)
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create output: {output_path}")
                writer.write(annotated)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    total_seconds = time.perf_counter() - total_start
    avg_ms = 1000 * inference_seconds / processed if processed else 0.0
    infer_fps = processed / inference_seconds if inference_seconds else 0.0
    e2e_fps = processed / total_seconds if total_seconds else 0.0
    print(f"Frames read: {frame_index}")
    print(f"Frames inferred: {processed}")
    print(f"Input size: {args.imgsz}")
    print(f"FP16 enabled: {use_half}")
    print(f"Average inference time: {avg_ms:.2f} ms/frame")
    print(f"Inference throughput: {infer_fps:.2f} FPS")
    print(f"End-to-end throughput: {e2e_fps:.2f} FPS")
    if output_path and writer is not None:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
