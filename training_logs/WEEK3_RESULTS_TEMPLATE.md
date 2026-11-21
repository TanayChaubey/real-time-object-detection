# Week 3 benchmark results

Recorded after running the benchmark locally. Values below are taken from `training_logs/week3_benchmark.json`.

- Machine: Apple MacBook Air M1
- GPU / accelerator: Apple MPS
- OS and Python version: Not recorded in benchmark output
- PyTorch version: 2.15.0
- Ultralytics version: 8.4.216
- Model checkpoint: `yolo11n.pt`
- Test source: `data/benchmark_images/`
- Number of images: 30
- Device: `mps`
- Warm-up images: 5
- Repeats: 3
- Timed inferences per configuration: 90
- Baseline input size: 640
- Optimized input size: 512
- FP16 enabled: No (both configurations)
- Frame stride: Not applicable to image-folder benchmark

| Metric | Baseline (640) | Optimized (512) |
|---|---:|---:|
| Mean latency (ms) | 26.20 | 13.56 |
| Median latency (ms) | 15.32 | 11.01 |
| P95 latency (ms) | 35.58 | 13.81 |
| Throughput (FPS) | 38.16 | 73.75 |
| Precision | Not measured | Not measured |
| Recall | Not measured | Not measured |
| mAP@50 | Not measured | Not measured |
| mAP@50–95 | Not measured | Not measured |

- Speedup: 1.93×
- Throughput increase: approximately 93.24%
- Mean latency reduction: approximately 48.25%
- Prediction agreement at IoU 0.50: 86.41%
- Visible quality differences: Not documented during this benchmark.
- Main failure cases: Not separately categorized.
- Conclusion: 512 input size improved measured inference throughput on this M1/MPS setup. Labelled validation is still needed to measure the accuracy trade-off.

**Important:** Prediction agreement compares optimized detections with baseline detections. It is not precision, recall, or mAP. The reported FPS is inference throughput from the image benchmark, not end-to-end video FPS.

Benchmark JSON: `training_logs/week3_benchmark.json`.
