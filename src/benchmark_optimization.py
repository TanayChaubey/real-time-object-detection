"""Compare baseline and optimized YOLO inference on the same image set.

The script reports latency/FPS and prediction agreement. Agreement is NOT
ground-truth accuracy or mAP; use labelled validation data for that.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark baseline vs optimized YOLO.")
    p.add_argument("--source", required=True, help="Folder containing benchmark images.")
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--device", default=None)
    p.add_argument("--baseline-imgsz", type=int, default=640)
    p.add_argument("--optimized-imgsz", type=int, default=512)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", default="training_logs/week3_benchmark.json")
    return p.parse_args()


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def unpack(result):
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    return [(int(c), float(s), b.tolist()) for c, s, b in zip(cls, conf, xyxy)]


def agreement(base, opt, threshold=0.5):
    if not base:
        return 1.0 if not opt else 0.0
    matched = set()
    hits = 0
    for cls_a, _, box_a in base:
        best_idx, best_iou = None, 0.0
        for idx, (cls_b, _, box_b) in enumerate(opt):
            if idx in matched or cls_a != cls_b:
                continue
            score = iou_xyxy(box_a, box_b)
            if score > best_iou:
                best_idx, best_iou = idx, score
        if best_idx is not None and best_iou >= threshold:
            matched.add(best_idx)
            hits += 1
    return hits / max(len(base), 1)


def run_config(model, images, imgsz, device, half, warmup, repeats, conf):
    kwargs = dict(imgsz=imgsz, device=device, half=half, conf=conf, verbose=False)
    for image in images[:min(warmup, len(images))]:
        model.predict(source=image, **kwargs)
    if torch.cuda.is_available() and str(device).startswith("0"):
        torch.cuda.synchronize()

    latencies = []
    predictions = []
    for _ in range(repeats):
        for image in images:
            start = time.perf_counter()
            result = model.predict(source=image, **kwargs)[0]
            if torch.cuda.is_available() and str(device).startswith("0"):
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            predictions.append(unpack(result))
    avg = statistics.mean(latencies)
    return {
        "image_size": imgsz,
        "half_precision": half,
        "images_in_timed_runs": len(latencies),
        "mean_latency_ms": avg * 1000,
        "median_latency_ms": statistics.median(latencies) * 1000,
        "p95_latency_ms": sorted(latencies)[int(0.95 * (len(latencies) - 1))] * 1000,
        "throughput_fps": 1 / avg if avg else 0,
        "predictions": predictions,
    }


def main():
    args = parse_args()
    folder = Path(args.source)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder}")
    images = sorted(p for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not images:
        raise ValueError("No supported images found in source folder.")
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be >= 1 and warmup must be >= 0.")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.model)
    baseline = run_config(model, images, args.baseline_imgsz, device, False,
                          args.warmup, args.repeats, args.conf)
    optimized_half = bool(torch.cuda.is_available() and str(device).startswith("0"))
    optimized = run_config(model, images, args.optimized_imgsz, device, optimized_half,
                           args.warmup, args.repeats, args.conf)

    base_preds = baseline.pop("predictions")
    opt_preds = optimized.pop("predictions")
    pairs = zip(base_preds, opt_preds)
    agreement_scores = [agreement(a, b) for a, b in pairs]
    agreement_mean = statistics.mean(agreement_scores) if agreement_scores else 0.0

    result = {
        "model": args.model,
        "device": device,
        "image_count": len(images),
        "warmup_images": min(args.warmup, len(images)),
        "repeats": args.repeats,
        "baseline": baseline,
        "optimized": optimized,
        "speedup": (baseline["mean_latency_ms"] / optimized["mean_latency_ms"]
                    if optimized["mean_latency_ms"] else None),
        "mean_prediction_agreement_iou50": agreement_mean,
        "interpretation": (
            "Prediction agreement compares optimized outputs with baseline outputs. "
            "It is not ground-truth precision, recall, or mAP."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved benchmark: {output}")


if __name__ == "__main__":
    main()
