# Training logs

Run the Week 2 training script to generate real logs and metrics. Ultralytics writes epoch-level loss and validation metrics, plots, and model checkpoints under runs/week2/yolo11n_coco128 by default. The script also writes week2_summary.json after final validation.

The generated run directory is intentionally ignored by Git because checkpoints and generated plots can be large. For submission, export or copy the relevant results.csv, week2_summary.json, results.png, and confusion_matrix.png into this folder after completing a real run. Do not fill this folder with estimated or fabricated metrics.

The COCO128 dataset is a small public sample for checking the training pipeline. Its score should not be presented as a reliable measure of generalization.
