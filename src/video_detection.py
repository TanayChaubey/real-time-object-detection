"""Run object detection on a video file or webcam stream."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.preprocessing import validate_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in a video file or camera stream."
    )
    parser.add_argument(
        "--source", required=True,
        help="Video path or camera index (use 0 for the default webcam).",
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", default="outputs")
    return parser.parse_args()


def get_video_source(source: str):
    """Convert a numeric source to a camera index."""
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive.")

    source = get_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    model = YOLO(args.model)
    writer = None
    output_path = None

    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(source, int):
            output_path = save_dir / "webcam_detected.mp4"
        else:
            output_path = save_dir / f"{Path(str(source)).stem}_detected.mp4"

    frame_count = 0
    model_call_time = 0.0
    total_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = validate_frame(frame)
            frame_start = time.perf_counter()

            options = {
                "source": frame,
                "conf": args.conf,
                "imgsz": args.imgsz,
                "verbose": False,
            }
            if args.device is not None:
                options["device"] = args.device

            result = model.predict(**options)[0]
            annotated = result.plot()

            elapsed = time.perf_counter() - frame_start
            model_call_time += elapsed
            frame_count += 1

            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated,
                f"Model call: {elapsed * 1000:.1f} ms | {fps:.1f} FPS",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (40, 220, 40),
                2,
                cv2.LINE_AA,
            )

            if args.save:
                if writer is None:
                    height, width = annotated.shape[:2]
                    source_fps = cap.get(cv2.CAP_PROP_FPS)
                    if source_fps <= 0 or source_fps > 120:
                        source_fps = 25.0
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create: {output_path}")
                writer.write(annotated)

            cv2.imshow("Real-Time Object Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    total_elapsed = time.perf_counter() - total_start
    average_ms = (
        model_call_time * 1000 / frame_count if frame_count else 0.0
    )
    end_to_end_fps = (
        frame_count / total_elapsed if total_elapsed > 0 else 0.0
    )

    print(f"Frames processed: {frame_count}")
    print(f"Average model-call time: {average_ms:.2f} ms/frame")
    print(f"End-to-end processing rate: {end_to_end_fps:.2f} FPS")
    if output_path is not None and writer is not None:
        print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
