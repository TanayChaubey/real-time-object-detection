"""Run pretrained YOLO object detection on a single image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.preprocessing import read_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in one image using a pretrained YOLO model."
    )
    parser.add_argument("--source", required=True, help="Path to an image.")
    parser.add_argument(
        "--model", default="yolo11n.pt", help="Model checkpoint to load."
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device", default=None,
        help="For example cpu or 0. Default lets Ultralytics choose.",
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1.")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive.")

    image = read_image(args.source)
    model = YOLO(args.model)

    options = {
        "source": image,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        options["device"] = args.device

    result = model.predict(**options)[0]
    annotated = result.plot()

    if args.save:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{Path(args.source).stem}_detected.jpg"
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save image: {output_path}")
        print(f"Saved: {output_path}")

    cv2.imshow("Object Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    detections = 0 if result.boxes is None else len(result.boxes)
    print(f"Detections: {detections}")


if __name__ == "__main__":
    main()
