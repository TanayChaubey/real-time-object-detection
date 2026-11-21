# Week 3 — Real-Time Performance Optimization

**Project:** Real-Time Object Detection  
**Baseline:** YOLO11n inference through the existing `src/video_detection.py`  
**Optimized path:** `src/optimized_video_detection.py`  
**Benchmark:** `src/benchmark_optimization.py`

> Benchmark status: source code and repeatable measurement procedure are provided. Hardware measurements have not yet been run in this environment. No speedup, FPS, latency, or accuracy result is invented. Run the benchmark and insert its JSON output before submitting measured claims.

## 1. Objective

The goal is to reduce inference latency and improve throughput while monitoring any change in detection outputs. The Week 2 YOLO11n checkpoint is reused so the comparison isolates inference-time settings rather than comparing different model architectures or training runs.

## 2. Bottleneck analysis

The main costs in a video detector are image resizing/letterboxing, tensor preparation and host-to-device transfer, neural-network forward execution, non-maximum suppression, drawing boxes, frame decoding, and display or video encoding. On smaller devices, model execution and per-frame preprocessing are usually important costs; display and encoding can also limit end-to-end throughput.

The original video script already loads the model once and reuses it across frames. The Week 3 implementation preserves this behavior rather than repeatedly loading weights.

## 3. Optimization techniques

### 3.1 Lower inference resolution

The baseline uses 640-pixel input. The optimized script defaults to 512 pixels. Fewer input pixels reduce the work in convolutional layers and intermediate feature maps. This can improve speed, but small or distant objects may become harder to detect. The setting is configurable so 640 and 512 can be compared on identical inputs.

### 3.2 FP16 inference on supported CUDA devices

The optimized video script exposes `--half`. It enables half-precision inference only when CUDA is available and a CUDA device is selected. FP16 can reduce memory traffic and improve throughput on compatible GPUs. It is not forced on CPU or Apple MPS because backend support and numerical behavior differ. On unsupported hardware, leave `--half` off.

### 3.3 Frame stride

The `--frame-stride N` option processes every Nth decoded frame. This reduces average compute demand for applications that can tolerate fewer temporal updates. It does not make an individual inference faster, and fast-moving objects may be missed between processed frames. Use 1 for every frame and report the chosen stride.

### 3.4 Bounded detections

The optimized path sets a configurable maximum number of detections per frame (default 100). This bounds post-processing and drawing work in crowded scenes. A low cap can discard valid objects, so increase it if the target scene is dense.

### 3.5 Reuse and measurement discipline

The model is loaded once outside the frame loop. Benchmarking includes warm-up iterations and reports mean, median, and 95th-percentile latency. CUDA synchronization is used around timed inference when benchmarking a CUDA device so asynchronous GPU work is included.

No structured pruning or quantization conversion is applied in this iteration. These techniques can alter model behavior or require backend-specific calibration/export; the selected changes are easier to reproduce and compare on the existing YOLO workflow.

## 4. Benchmark procedure

Install project dependencies:

```bash
pip install -r requirements.txt
```

Place a representative, fixed set of images in a folder, for example `data/benchmark_images/`. Use the same images, model checkpoint, confidence threshold, and device for both configurations.

Run the baseline video implementation:

```bash
python -m src.video_detection --source path/to/test_video.mp4 --imgsz 640 --conf 0.25
```

Run the optimized video implementation:

```bash
python -m src.optimized_video_detection --source path/to/test_video.mp4 --imgsz 512 --conf 0.25 --no-display
```

On a CUDA GPU, optionally enable FP16:

```bash
python -m src.optimized_video_detection --source path/to/test_video.mp4 --imgsz 512 --half --no-display
```

Benchmark baseline versus optimized settings on an image folder:

```bash
python -m src.benchmark_optimization --source data/benchmark_images --model yolo11n.pt --baseline-imgsz 640 --optimized-imgsz 512 --warmup 5 --repeats 3
```

The benchmark automatically enables FP16 for the optimized configuration only when CUDA is available. Use a fixed local copy of the same images and avoid mixing different image sets between runs.

## 5. Metrics

- **Mean latency (ms/image):** average measured inference time.
- **Median latency:** central latency, less sensitive to outliers.
- **P95 latency:** latency below which 95% of timed calls fall.
- **Throughput (FPS):** reciprocal of mean latency for the timed inference call.
- **Speedup:** baseline mean latency divided by optimized mean latency.
- **Prediction agreement:** proportion of baseline detections matched by optimized detections of the same class at IoU ≥ 0.50. This is a consistency check, not ground-truth accuracy.
- **Precision, recall, mAP:** must be evaluated on labelled validation data to assess accuracy impact. The benchmark script does not claim these metrics.

## 6. Results table — populate from a real run

| Metric | Baseline (640, FP32) | Optimized (512, device-dependent FP16) |
|---|---:|---:|
| Mean latency (ms) | Run benchmark | Run benchmark |
| Median latency (ms) | Run benchmark | Run benchmark |
| P95 latency (ms) | Run benchmark | Run benchmark |
| Throughput (FPS) | Run benchmark | Run benchmark |
| Speedup | 1.00× reference | Read JSON output |
| Prediction agreement (IoU 0.50) | Reference | Read JSON output |
| Precision / Recall / mAP | Evaluate labelled set | Evaluate labelled set |

The benchmark writes `training_logs/week3_benchmark.json`. Copy its actual values into this table. For a fair accuracy comparison, validate both settings on the same labelled split and compare precision, recall, mAP@50, and mAP@50–95. Prediction agreement alone is not a substitute.

## 7. Trade-offs and limitations

Reducing image size can reduce recall for small objects. FP16 may introduce small numerical differences and is hardware-dependent. Frame skipping reduces temporal sampling and may miss brief events. A maximum-detection cap can truncate crowded scenes. FPS measured for a single image inference call is not necessarily end-to-end camera FPS; report both when possible and name the hardware, software versions, device, image size, and video codec.

## 8. Further work

1. Export to ONNX and benchmark an optimized runtime such as ONNX Runtime on the target hardware.
2. Evaluate TensorRT FP16/INT8 on supported NVIDIA hardware, using representative calibration data for INT8.
3. Apply structured channel pruning only with fine-tuning and a measured accuracy comparison.
4. Benchmark different YOLO model sizes and input resolutions on the deployment device.
5. Use asynchronous frame capture and a bounded queue if camera decoding blocks inference.
6. Add object tracking so detections need not be recomputed on every frame, while validating tracking drift and ID switches.
7. Profile preprocessing, inference, NMS, drawing, and encoding separately before optimizing further.

## 9. Conclusion

Week 3 adds a separate optimized video inference path and a repeatable baseline-versus-optimized benchmark while preserving the original implementation as a reference. The initial optimization focuses on lower input resolution, optional CUDA FP16, configurable frame stride, bounded detections, and warm-up-aware latency measurement. Actual speed and accuracy trade-offs must be established by running the benchmark on the intended hardware and evaluating both configurations against labelled data.
