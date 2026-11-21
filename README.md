# Real-Time Object Detection

A Python object-detection project using YOLO and OpenCV. It supports image, video, and webcam inference, plus a reproducible Week 2 model fine-tuning workflow.

## Project layout

    real-time-object-detection/
    ├── README.md
    ├── requirements.txt
    ├── docs/
    │   ├── week1_algorithm_design.md
    │   └── week2_training_report.md
    ├── notebooks/
    │   ├── README.md
    │   └── week2_yolo_training.ipynb
    ├── src/
    │   ├── detection.py
    │   ├── preprocessing.py
    │   ├── train.py
    │   └── video_detection.py
    └── training_logs/
        ├── README.md
        └── RESULTS_TEMPLATE.md

## Setup

Python 3.10 or newer is recommended.

    git clone https://github.com/TanayChaubey/real-time-object-detection.git
    cd real-time-object-detection
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

On Windows, activate the environment with .venv\Scripts\activate.

## Inference

Image:

    python -m src.detection --source path/to/image.jpg --save

Video:

    python -m src.video_detection --source path/to/video.mp4 --save

Webcam:

    python -m src.video_detection --source 0

Press q to stop video or webcam processing.

## Week 2 — Train the detector

The training baseline fine-tunes pretrained YOLO11n weights on the public COCO128 sample dataset using PyTorch through Ultralytics. The first run downloads the checkpoint and dataset.

    python -m src.train

Optional configuration:

    python -m src.train --epochs 50 --imgsz 640 --batch 8 --seed 42

For a constrained machine, lower the batch size:

    python -m src.train --epochs 20 --imgsz 416 --batch 4

For a short CPU smoke test:

    python -m src.train --device cpu --epochs 3 --batch 2

The full documented workflow is in notebooks/week2_yolo_training.ipynb. The methodology, parameters, evaluation metrics, limitations, and improvement plan are in docs/week2_training_report.md.

## Training outputs

The script writes epoch history, plots, checkpoints, and a JSON summary under runs/week2/yolo11n_coco128 by default. Important files include:

- results.csv — training and validation history.
- results.png — loss and metric curves.
- confusion_matrix.png — class confusion visualization, when generated.
- weights/best.pt — best checkpoint.
- week2_summary.json — actual validation metrics and experiment settings.

Generated runs and model weights are ignored by Git to avoid large accidental commits. Copy the actual small logs/plots into training_logs if needed for submission.

## Dataset and metric caveat

COCO128 is a small public sample dataset used to verify the training pipeline. It is not sufficient to claim production-level accuracy or generalization. The report deliberately contains no fabricated metric values. Run training and validation, then use the generated results.csv and week2_summary.json to complete the results section.

For object detection, report precision, recall, mAP@50, and mAP@50–95. Also inspect box/classification/DFL losses, per-class behavior, false positives, false negatives, and runtime on identified hardware.

## Current status

- [x] Week 1 algorithm and preprocessing strategy.
- [x] Image, video, and webcam inference scripts.
- [x] Week 2 YOLO11n training script.
- [x] Documented Week 2 notebook and report.
- [ ] Execute the training run and preserve genuine logs/plots.
- [ ] Evaluate on a larger independent or target-domain dataset.
- [ ] Compare model sizes and augmentation settings.

## Week 3 — Real-time inference optimization

The original `src/video_detection.py` remains the baseline. The optimized path is in `src/optimized_video_detection.py` and supports a smaller input size (default 512), optional FP16 on compatible CUDA devices, frame stride, and a configurable maximum detections per frame.

Run optimized video inference:

    python -m src.optimized_video_detection --source path/to/video.mp4 --imgsz 512 --no-display

On a supported CUDA GPU, add `--half` to enable FP16. Do not expect FP16 acceleration on CPU or Apple MPS.

Compare inference settings on a fixed image folder:

    python -m src.benchmark_optimization --source data/benchmark_images --baseline-imgsz 640 --optimized-imgsz 512 --warmup 5 --repeats 3

The benchmark writes `training_logs/week3_benchmark.json` with mean/median/P95 latency, throughput, speedup, and prediction agreement. Prediction agreement is not ground-truth accuracy. Evaluate precision, recall, and mAP on a labelled validation set before claiming no accuracy loss.

See `docs/week3_optimization_report.md` and `training_logs/WEEK3_RESULTS_TEMPLATE.md` for the methodology and results record. Actual hardware measurements must be collected locally; none are fabricated in the report.
