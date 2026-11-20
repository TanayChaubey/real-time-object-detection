"""Fine-tune a pretrained YOLO detector on the COCO128 sample dataset.

The dataset is downloaded by Ultralytics on first use. COCO128 is a small
pipeline/demo dataset; do not treat its metrics as a production benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a YOLO11n object detector."
    )
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--data", default="coco128.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/week2")
    parser.add_argument("--name", default="yolo11n_coco128")
    parser.add_argument("--device", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("epochs, batch and image size must be positive.")
    set_seed(args.seed)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device argument:", args.device or "Ultralytics automatic selection")
    print("Dataset:", args.data)
    print("Starting training...")

    model = YOLO(args.model)
    train_options = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "plots": True,
        "save": True,
        "verbose": True,
    }
    if args.device:
        train_options["device"] = args.device
    if args.resume:
        train_options["resume"] = True

    start = time.perf_counter()
    train_result = model.train(**train_options)
    training_seconds = time.perf_counter() - start

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Expected best checkpoint at {best_weights}")

    print("Training finished. Evaluating best checkpoint...")
    best_model = YOLO(str(best_weights))
    val_result = best_model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        plots=True,
        project=str(run_dir),
        name="final_validation",
        exist_ok=True,
    )

    metrics = {}
    if hasattr(val_result, "results_dict"):
        for key, value in val_result.results_dict.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = str(value)

    summary = {
        "model": args.model,
        "dataset": args.data,
        "epochs_requested": args.epochs,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "seed": args.seed,
        "device": args.device or "auto",
        "training_seconds": round(training_seconds, 2),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights) if last_weights.exists() else None,
        "validation_metrics": metrics,
        "note": "COCO128 is a small sample dataset; results are not a production benchmark.",
    }

    summary_path = run_dir / "week2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        print(f"Epoch history saved by Ultralytics: {csv_path}")
        print(f"Epochs recorded: {len(rows)}")

    print(json.dumps(summary, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
