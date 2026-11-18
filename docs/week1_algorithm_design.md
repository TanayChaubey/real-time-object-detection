# Week 1 — Algorithm Design and Data Preprocessing Strategy

**Project:** Real-Time Object Detection for Urban / Road-Scene Imagery  
**Status:** Design and initial inference baseline

> This document records the proposed approach. It does not claim that a target-domain model has been trained or that benchmark accuracy/FPS results have been achieved.

## 1. Problem definition

The system receives an image, video file, or camera frame and predicts object classes, confidence scores, and bounding boxes. The first use case is road-scene awareness, with common classes such as person, bicycle, car, motorcycle, bus, truck, traffic light, and stop sign.

The output is a set of detections in the form `(x1, y1, x2, y2, class, confidence)`, together with an annotated frame. Tracking, lane detection, depth estimation, and safety-critical decisions are outside the initial scope.

## 2. Algorithm choice

A one-stage YOLO-family detector is the baseline because it predicts boxes and classes in a single network pass and has a practical speed/accuracy trade-off. The initial checkpoint is `yolo11n.pt`, a small pretrained model intended to keep the first experiments lightweight. Model version, checkpoint source, license, and runtime compatibility should be confirmed before deployment.

The system uses the Ultralytics high-level prediction interface. That interface handles model-specific input resizing/letterboxing, tensor conversion, inference, confidence filtering, and NMS. The project should not repeat these operations manually unless switching to a lower-level model API.

### Inference stages

1. Read and validate the input.
2. Pass the frame to the model prediction API with configured confidence and image size.
3. Decode predictions and rely on the model package's configured post-processing/NMS.
4. Render boxes and labels.
5. Save or display the result.
6. Record frame time and throughput for later benchmarking.

## 3. Dataset strategy

For a generic baseline, use a public object-detection dataset such as COCO. For a road-scene-specific model, consider BDD100K or a permitted, properly annotated local dataset. Confirm license, use restrictions, and privacy requirements before using any dataset.

Use a detection annotation format supported by the selected training library. Validate that class IDs are known, boxes have positive width/height and valid coordinates, and image/annotation pairs are complete. Review a sample of labels manually.

Split data by video sequence, camera, or collection session where possible. A random split of adjacent frames can put near-identical images in both training and test sets and make evaluation misleading. A starting split could be 70% train, 15% validation, and 15% test, unless an official benchmark split is required.

## 4. Preprocessing plan

| Step | Planned operation | Reason |
|---|---|---|
| File validation | Check decode success, shape, channels, and label pairing | Catch broken inputs early |
| Color order | Use the channel order expected by the inference API | Avoid BGR/RGB mistakes |
| Resize | Model-specific resize, initially around 640 px input | Standard model input size |
| Letterboxing | Preserve aspect ratio and pad as required by the model | Avoid geometric distortion |
| Normalization | Apply the exact model input scaling | Match pretrained model conventions |
| Label transform | Apply the same geometric transform to bounding boxes | Keep boxes aligned |
| Noise handling | Start without denoising; test mild filtering only if noise is visible | Avoid removing small-object details |
| Duplicate audit | Check repeated images and adjacent frames | Reduce leakage and redundancy |

The high-level Ultralytics API owns its inference preprocessing. The standalone `resize_with_letterbox` helper is included for future custom preprocessing or inspection, not as an extra step before the high-level API.

## 5. Augmentation plan

Augmentation is for training only. Validation and test data should use deterministic preprocessing.

- Horizontal flip, where orientation is not semantically meaningful.
- Mild scale and translation changes.
- Small rotations only when realistic for the camera.
- Brightness, contrast, and mild HSV changes for lighting variation.
- Detection-aware crops that update or remove boxes consistently.
- Optional blur/noise at realistic strengths.
- Mosaic or mixup only if supported by the selected training recipe and checked visually.

Every geometric transform must update the bounding-box coordinates. Inspect transformed images and labels before training. Avoid unrealistic transformations that change the scene's meaning.

## 6. Inference flow

```text
Image / video / camera frame
             |
       Decode + validate
             |
      Is the frame valid?
        /           \
      No             Yes
      |               |
   Log/skip       Model API
                      |
                  Inference
                      |
              Confidence + NMS
              (inside API)
                      |
                Draw detections
                      |
               Display / save
                      |
                Log timing
```

## 7. Training and evaluation workflow

```text
Collect images and labels
          |
     Audit annotations
          |
  Group-aware data split
          |
  Augment training samples
          |
  Load pretrained weights
          |
      Fine-tune model
          |
  Evaluate on validation set
          |
    Tune model/settings
          |
   Freeze chosen settings
          |
   Evaluate test set once
          |
 Report metrics and limitations
```

## 8. Pseudocode

### Image/video inference

```text
FUNCTION detect_frame(frame, model, config):
    IF frame is empty or invalid:
        RETURN error

    predictions = model.predict(
        frame,
        confidence=config.confidence,
        image_size=config.image_size
    )

    detections = predictions[0]
    annotated_frame = draw_boxes_and_labels(frame, detections)

    RETURN detections, annotated_frame
```

### Dataset preparation

```text
FUNCTION prepare_dataset(images, annotations):
    samples = pair_images_and_labels(images, annotations)
    valid_samples = []

    FOR sample IN samples:
        IF image cannot be decoded:
            log and continue

        IF class IDs or boxes are invalid:
            log and continue

        valid_samples.append(sample)

    groups = group_by_sequence_camera_or_session(valid_samples)
    train, validation, test = split_groups(groups, 70%, 15%, 15%)

    assert no group overlaps across splits
    RETURN train, validation, test
```

## 9. Evaluation metrics

- **Precision:** TP / (TP + FP), the fraction of predicted detections that are correct.
- **Recall:** TP / (TP + FN), the fraction of ground-truth objects found.
- **IoU:** intersection area divided by union area for predicted and ground-truth boxes.
- **AP:** area under the precision-recall curve for one class at a defined IoU threshold.
- **mAP@50:** mean AP at IoU 0.50.
- **mAP@50–95:** mean AP averaged over IoU thresholds from 0.50 through 0.95 in 0.05 steps.
- **Latency:** time per frame, ideally separated into preprocessing, inference, and post-processing.
- **FPS:** processed frames per second. Report hardware, image size, batch size, and precision mode.

No mAP or FPS score is reported yet. Those values require a labelled evaluation set and a known test device.

## 10. Design rationale and trade-offs

| Choice | Benefit | Trade-off |
|---|---|---|
| One-stage YOLO model | Practical real-time inference | Smaller model can miss difficult/small objects |
| Pretrained weights | Faster starting point and less labelled data needed | May not match local camera domain |
| Letterbox resize | Preserves object geometry | Padding uses part of the input canvas |
| Moderate augmentation | Adds useful scene variation | Over-augmentation can make examples unrealistic |
| Group-aware split | Reduces near-duplicate leakage | Needs source/sequence metadata |
| Start with small model | Lower compute and memory demand | Larger model may be needed for accuracy |

## 11. Risks and limitations

- Public data may not match the internship's target environment.
- Small, occluded, blurred, or night-time objects may be missed.
- Class imbalance can hide poor per-class performance.
- Annotation quality directly affects training and evaluation.
- Runtime depends on hardware, video decoding, and display overhead.
- A benchmark result does not establish that a model is safe for driving decisions.

## 12. Next implementation steps

1. Confirm target classes, input source, and available hardware.
2. Run the baseline scripts on representative images and video.
3. Select and audit a permitted dataset.
4. Create annotation checks and group-aware split manifests.
5. Fine-tune only if target-domain labels are available.
6. Compare model sizes and thresholds on validation data.
7. Run final test evaluation and report per-class metrics and latency.

## References

- Ultralytics documentation: https://docs.ultralytics.com/
- COCO dataset: https://cocodataset.org/
- BDD100K: https://www.vis.xyz/bdd100k/
- OpenCV documentation: https://docs.opencv.org/
- PyTorch documentation: https://pytorch.org/docs/stable/
