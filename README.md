# Real-Time Object Detection

A small Python project for detecting common objects in images and video using a pretrained YOLO model. OpenCV handles image/video input and display, while Ultralytics provides the model interface.

This repository is a practical starting point for the Computer Vision Specialist Internship. The first milestone focuses on a clean inference pipeline and the algorithm/data strategy. It does **not** claim custom training or measured benchmark results yet.

## What it does

- Runs object detection on a single image.
- Processes a video file or webcam stream frame by frame.
- Lets you change the model, confidence threshold, and inference image size.
- Saves annotated image/video outputs when requested.
- Keeps image preparation in a separate module so it is easier to test and improve.

## Project layout

```
real-time-object-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── week1_algorithm_design.md
├── src/
│   ├── __init__.py
│   ├── detection.py
│   ├── preprocessing.py
│   └── video_detection.py
├── notebooks/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Setup

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/TanayChaubey/real-time-object-detection.git
cd real-time-object-detection

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The first run may download the selected pretrained model weights. Check the model's license and terms for your intended use.

## Run detection

### Image

```bash
python -m src.detection --source path/to/image.jpg --save
```

### Video file

```bash
python -m src.video_detection --source path/to/video.mp4 --save
```

### Webcam

```bash
python -m src.video_detection --source 0
```

Press `q` in the display window to stop video or webcam processing.

## Useful options

Both scripts accept common options:

- `--model`: model checkpoint (default: `yolo11n.pt`).
- `--conf`: confidence threshold (default: `0.25`).
- `--imgsz`: inference image size (default: `640`).
- `--device`: device such as `cpu` or `0` (default: automatic selection).
- `--save-dir`: output directory (default: `outputs`).

Example:

```bash
python -m src.detection --source road.jpg --conf 0.35 --imgsz 640 --save
```

## Pipeline at a glance

1. Read and validate the source.
2. Pass the frame to the model API.
3. Run model inference.
4. Apply confidence filtering and the model package's post-processing/NMS.
5. Render labels and boxes, then optionally save the result.

The Ultralytics prediction API performs model-specific preprocessing and post-processing internally. The separate preprocessing module contains reusable validation and image utility functions. Do not normalize or letterbox a frame manually and then pass it into the high-level API unless that API explicitly expects such input.

## Current status

- [x] Week 1 algorithm and data strategy documented.
- [x] Initial image inference script.
- [x] Initial video/webcam inference script.
- [ ] Test with the internship's target dataset.
- [ ] Add annotation validation and group-aware dataset splitting.
- [ ] Fine-tune a model if a labelled target-domain dataset is provided.
- [ ] Record mAP, latency, FPS, hardware, and experiment settings.
- [ ] Add real sample outputs after running the pipeline.

## Evaluation plan

For a labelled detection dataset, report per-class Precision, Recall, AP, mAP@50, and mAP@50–95. For runtime, measure preprocessing, inference, post-processing, and end-to-end latency separately, then report FPS with hardware, image size, batch size, and precision mode. No numeric accuracy or FPS result is claimed in this initial milestone.

## Notes

- A pretrained COCO model is a generic starting point; it may not perform well on every local camera or road-scene condition.
- Split video data by sequence/camera/session where possible to reduce near-duplicate leakage.
- This is an educational prototype, not a safety-critical driving system.
