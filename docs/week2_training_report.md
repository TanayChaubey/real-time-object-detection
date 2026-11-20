# Week 2 — Deep Learning Model Implementation and Training

**Project:** Real-Time Object Detection  
**Framework:** PyTorch through Ultralytics  
**Model:** YOLO11n pretrained checkpoint  
**Dataset:** COCO128 public sample dataset  
**Status:** Training implementation and experiment protocol

> Numeric results are not fabricated. Run the training script and use results.csv and week2_summary.json to complete the results section.

## 1. Objective

Fine-tune a deep-learning object detector and evaluate detection quality and runtime. This continues the Week 1 object-detection project. The detector predicts a class, confidence score, and bounding box for each object.

The baseline uses pretrained YOLO11n weights and COCO128 to verify data loading, training, checkpoint saving, and validation. COCO128 is a small sample suitable for learning and pipeline validation, not production benchmarking.

## 2. Model architecture

YOLO11n is a compact one-stage object detector. Its high-level architecture contains a convolutional feature-extraction backbone, a feature aggregation stage that combines information at multiple resolutions, and a detection head that predicts bounding boxes and classes at multiple scales. The training objective combines localization, classification, and distribution-based box regression terms as implemented by the installed Ultralytics version.

The nano variant reduces compute and memory demand. Pretrained weights provide a useful initialization rather than training all parameters from random values.

## 3. Dataset

The experiment uses COCO128, a small sample derived from COCO and distributed through the Ultralytics dataset configuration. The script uses the built-in coco128.yaml configuration; Ultralytics downloads and prepares the dataset on first use.

This small sample is useful for exercising the pipeline but is not a large independent test set. A stronger follow-up should use a separate official split or a target-domain road-scene dataset such as BDD100K after checking its terms and annotation format.

## 4. Preprocessing and augmentation

Ultralytics handles image decoding, label parsing, resize/letterbox, pixel conversion, normalization, bounding-box coordinate transformations, and training-time detection augmentation.

The baseline keeps the library defaults for reproducibility. A controlled follow-up can tune horizontal flip, scale, translation, HSV/color changes, mosaic, and mixup. Geometric augmentation must transform boxes with images. Validation should remain deterministic and must not use random training augmentation.

## 5. Training configuration

| Parameter | Baseline value | Reason |
|---|---:|---|
| Model | yolo11n.pt | Small pretrained detector |
| Dataset | coco128.yaml | Public sample dataset |
| Epochs | 30 | Initial learning run |
| Image size | 640 | Common detector input size |
| Batch size | 8 | Moderate memory demand |
| Optimizer | Ultralytics default | Keep baseline recipe consistent |
| Learning rate | Ultralytics default | Tune after baseline |
| Early stopping patience | 10 | Stop when validation does not improve |
| Random seed | 42 | Improve repeatability |
| Workers | 2 | Conservative data-loader parallelism |
| Device | automatic | Use available accelerator or CPU |

Record actual hardware and software versions at run time.

## 6. How to run

Install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run training from the repository root:

    python -m src.train

Optional settings:

    python -m src.train --epochs 50 --imgsz 640 --batch 8 --seed 42

For a constrained machine:

    python -m src.train --epochs 20 --imgsz 416 --batch 4

For a short CPU smoke test:

    python -m src.train --device cpu --epochs 3 --batch 2

The first run downloads the pretrained checkpoint and COCO128 data, so internet access and disk space are required. A GPU is recommended but not mandatory.

## 7. Training logs and artifacts

The run directory is created under runs/week2/yolo11n_coco128 by default. Ultralytics records epoch-level losses and metrics in results.csv, diagnostic plots, and checkpoints. The script evaluates best.pt and writes week2_summary.json containing configuration, elapsed training time, and validation metrics.

Preserve these artifacts:

- results.csv — epoch-level loss and validation history.
- results.png — training curves.
- confusion_matrix.png — class confusion summary, when generated.
- weights/best.pt — best checkpoint.
- week2_summary.json — final validation metrics and run configuration.

Generated weights and run artifacts are excluded from Git because they can be large. Copy selected small logs and plots to training_logs after a real run if required by the submission portal.

## 8. Evaluation metrics

Object detection is not evaluated with classification accuracy alone.

- **Precision:** proportion of predicted detections that are correct.
- **Recall:** proportion of ground-truth objects detected.
- **mAP@50:** mean average precision at IoU 0.50.
- **mAP@50–95:** mean AP averaged over IoU thresholds 0.50 to 0.95.
- **Box loss:** localization-related training loss.
- **Classification loss:** class prediction training loss.
- **DFL loss:** distribution focal loss used by the detector's box regression formulation.
- **Latency and FPS:** runtime measures that must be measured on named hardware.

A falling training loss alone does not prove good generalization. Compare training and validation behavior, inspect false positives and false negatives, and review class-wise performance.

### Results to fill after execution

| Measure | Actual result |
|---|---|
| Training device / hardware | Fill from your machine |
| Epochs completed | Read results.csv |
| Final box loss | Read results.csv |
| Final classification loss | Read results.csv |
| Precision | Read week2_summary.json |
| Recall | Read week2_summary.json |
| mAP@50 | Read week2_summary.json |
| mAP@50–95 | Read week2_summary.json |
| Training duration | Read week2_summary.json |

Do not submit blank or invented results as measured values. If the run fails or is interrupted, document the error and completed epochs.

## 9. Challenges and troubleshooting

- **First-run downloads:** model and dataset downloads require internet access and disk space.
- **Out-of-memory errors:** reduce batch size first, then image size.
- **Slow CPU training:** run a short smoke test before the full experiment.
- **Dataset path errors:** use built-in coco128.yaml for the baseline rather than a custom path to data that has not been downloaded.
- **Overfitting:** the small dataset can produce unstable validation metrics. Use more varied labelled data and an independent split.
- **Class imbalance:** report per-class AP and inspect the confusion matrix, not only aggregate mAP.

## 10. Improvements

1. Replace the small sample with a larger, legally usable detection dataset.
2. Use sequence-aware train/validation/test splits for video-derived images.
3. Compare nano and small model variants under identical settings.
4. Tune image size, batch size, learning rate, and augmentation one variable at a time.
5. Evaluate low-light, motion-blur, occlusion, and small-object cases.
6. Measure end-to-end latency on intended deployment hardware.
7. Add tracking only after evaluating the detector baseline.

## 11. Conclusion

This implementation provides a reproducible starting point for practical detector training: it loads pretrained weights, downloads a public sample dataset, fine-tunes the detector, validates the best checkpoint, and saves machine-readable metrics and training history. The next necessary step is to execute the experiment and attach genuine logs and plots. COCO128 is a pipeline demonstration, not evidence of production readiness.
