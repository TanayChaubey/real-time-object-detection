# Week 3 — Real-Time Performance Optimization

**Project:** Real-Time Object Detection  
**Baseline:** YOLO11n inference at 640-pixel input resolution  
**Optimized path:** `src/optimized_video_detection.py`  
**Benchmark:** `src/benchmark_optimization.py`  
**Test hardware:** Apple MacBook Air M1  
**Execution device:** Apple MPS  
**Benchmark status:** Completed on 30 images, with 5 warm-up images and 3 timed repeats.

## 1. Objective

The goal is to reduce inference latency and improve throughput while monitoring changes in model outputs. The same YOLO11n checkpoint and image set were used for the baseline and optimized configurations so the comparison focuses on inference-time settings rather than different model architectures or training runs.

## 2. Bottleneck analysis

The main costs in a video detector are image resizing/letterboxing, tensor preparation and host-to-device transfer, neural-network forward execution, non-maximum suppression, drawing boxes, frame decoding, and display or video encoding. On smaller devices, model execution and per-frame preprocessing are important costs; display and encoding can also limit end-to-end throughput.

The model is loaded once and reused across frames. The Week 3 implementation preserves this behavior rather than repeatedly loading weights.

## 3. Optimization techniques

### 3.1 Lower inference resolution

The baseline uses 640-pixel input. The optimized benchmark uses 512 pixels. Fewer input pixels reduce the work in convolutional layers and intermediate feature maps. This improved measured inference speed in this experiment, but small or distant objects may become harder to detect. The setting remains configurable so 640 and 512 can be compared on identical inputs.

### 3.2 FP16 inference on supported CUDA devices

The optimized video script exposes `--half`. It enables half-precision inference only when CUDA is available and a CUDA device is selected. FP16 can reduce memory traffic and improve throughput on compatible GPUs. It is not forced on CPU or Apple MPS because backend support and numerical behavior differ.

**For this benchmark, FP16 was disabled.** Both configurations ran without half precision on Apple MPS. Therefore, the measured improvement should be attributed to the input-resolution change and benchmark configuration, not FP16.

### 3.3 Frame stride

The `--frame-stride N` option processes every Nth decoded frame. This reduces average compute demand for applications that can tolerate fewer temporal updates. It does not make an individual inference faster, and fast-moving objects may be missed between processed frames. Use 1 for every frame and report the chosen stride.

### 3.4 Bounded detections

The optimized path sets a configurable maximum number of detections per frame (default 100). This bounds post-processing and drawing work in crowded scenes. A low cap can discard valid objects, so increase it if the target scene is dense.

### 3.5 Reuse and measurement discipline

The model is loaded once outside the frame loop. The benchmark uses warm-up iterations and reports mean, median, and 95th-percentile latency. The same 30 images were evaluated in both configurations, with 5 warm-up images and 3 timed repeats (90 timed image inferences per configuration).

## 4. Benchmark procedure

Install project dependencies:

```bash
pip install -r requirements.txt
```

Place a representative, fixed set of images in `data/benchmark_images/`. Use the same images, model checkpoint, confidence threshold, and device for both configurations.

The benchmark was run on the Apple M1 using MPS:

```bash
python -m src.benchmark_optimization --source data/benchmark_images --device mps --baseline-imgsz 640 --optimized-imgsz 512 --warmup 5 --repeats 3
```

The benchmark output is saved to `training_logs/week3_benchmark.json`.

The video inference script can be run separately:

```bash
python -m src.optimized_video_detection --source path/to/test_video.mp4 --device mps --imgsz 512
```

The image-folder benchmark measures model inference latency per image. It is not a measurement of full camera-to-display video FPS.

## 5. Metrics

- **Mean latency (ms/image):** average measured inference time.
- **Median latency:** central latency, less sensitive to outliers.
- **P95 latency:** latency below which 95% of timed calls fall.
- **Throughput (FPS):** reciprocal of mean latency for the timed inference call.
- **Speedup:** baseline mean latency divided by optimized mean latency.
- **Prediction agreement:** proportion of baseline detections matched by optimized detections of the same class at IoU ≥ 0.50. This is a consistency check, not ground-truth accuracy.
- **Precision, recall, mAP:** must be evaluated on labelled validation data to assess accuracy impact. The benchmark script does not calculate these metrics.

## 6. Measured results

**Test configuration**

| Parameter | Value |
|---|---|
| Model | YOLO11n (`yolo11n.pt`) |
| Machine | Apple MacBook Air M1 |
| Device | Apple MPS |
| Images evaluated | 30 |
| Warm-up images | 5 |
| Repeats | 3 |
| Timed inferences per configuration | 90 |
| Baseline input size | 640 |
| Optimized input size | 512 |
| FP16 | Disabled for both configurations |

**Benchmark measurements**

| Metric | Baseline (640, FP32) | Optimized (512, FP32) |
|---|---:|---:|
| Mean latency (ms) | 26.20 | 13.56 |
| Median latency (ms) | 15.32 | 11.01 |
| P95 latency (ms) | 35.58 | 13.81 |
| Throughput (FPS) | 38.16 | 73.75 |

- **Measured speedup:** 1.93×.
- **Throughput increase:** approximately 93.24%.
- **Mean latency reduction:** approximately 48.25%.
- **P95 latency reduction:** approximately 61.19%.
- **Prediction agreement at IoU 0.50:** 86.41%.

The optimized configuration produced substantially lower measured latency and higher inference throughput on this test set. Prediction agreement of 86.41% indicates that some detections differed between the two input sizes. This is expected to be monitored because reducing input resolution can affect smaller or less prominent objects.

These figures describe this particular machine, model, image set, and benchmark run. They should not be generalized to other hardware or described as end-to-end video FPS.

**Accuracy limitation:** Precision, recall, mAP@50, and mAP@50–95 were not measured because this benchmark run did not use ground-truth labels. Do not report the prediction-agreement score as model accuracy.

## 7. Trade-offs and limitations

Reducing image size can reduce recall for small objects. The measured 86.41% prediction agreement confirms that the optimized outputs were not identical to baseline outputs. FP16 was not used in this MPS experiment. Frame skipping reduces temporal sampling and may miss brief events. A maximum-detection cap can truncate crowded scenes. FPS measured for a single image inference call is not necessarily end-to-end camera FPS; report both when possible and name the hardware, software versions, device, image size, and video codec.

The benchmark output also displayed a deprecation warning related to the `half` argument. It did not prevent completion, and half precision was disabled in the recorded MPS measurements.

## 8. Further work

1. Evaluate both input sizes on a labelled validation set and compare precision, recall, mAP@50, and mAP@50–95.
2. Run the video pipeline on a fixed test video and record end-to-end FPS separately from model inference FPS.
3. Export to ONNX and benchmark an optimized runtime such as ONNX Runtime on the target hardware.
4. Evaluate TensorRT FP16/INT8 on supported NVIDIA hardware, using representative calibration data for INT8.
5. Benchmark different YOLO model sizes and input resolutions on the deployment device.
6. Use asynchronous frame capture and a bounded queue if camera decoding blocks inference.
7. Profile preprocessing, inference, NMS, drawing, and encoding separately before optimizing further.

## 9. Conclusion

The Week 3 benchmark on an Apple MacBook Air M1 using Apple MPS measured 38.16 FPS at 640 input resolution and 73.75 FPS at 512 input resolution. This corresponds to a 1.93× throughput speedup and a 48.25% reduction in mean inference latency for the tested image set. The optimized configuration had 86.41% prediction agreement with the baseline at IoU 0.50, so the speed improvement comes with some change in detections.

The results support using 512 input resolution as a promising faster configuration for this test setup, but a labelled validation run is still required before concluding that detection accuracy is acceptable for deployment.
