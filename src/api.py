"""REST API that exposes the existing YOLO detector to other applications."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from ultralytics import YOLO

app = FastAPI(
    title="YOLO11n Visual Inspection API",
    description="Image and video object detection service.",
    version="1.0.0",
)

_model: YOLO | None = None


def get_model() -> YOLO:
    """Load the detector once per API process."""
    global _model
    if _model is None:
        _model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))
    return _model


def _serialize_detections(result: Any) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = boxes.xyxy.cpu().tolist()
    confidences = boxes.conf.cpu().tolist()
    class_ids = boxes.cls.cpu().tolist()
    names = result.names
    detections = []
    for xyxy, confidence, class_id in zip(coordinates, confidences, class_ids):
        class_id = int(class_id)
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        detections.append({
            "class_id": class_id,
            "label": str(label),
            "confidence": round(float(confidence), 5),
            "xyxy": [round(float(value), 2) for value in xyxy],
        })
    return detections


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "yolo11n-visual-inspection"}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> dict[str, Any]:
    """Detect objects in an uploaded image and return JSON plus annotated image."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="File is not a decodable image.")

    try:
        result = get_model().predict(source=image, conf=0.25, imgsz=640, verbose=False)[0]
        annotated = result.plot()
        success, encoded = cv2.imencode(".jpg", annotated)
        if not success:
            raise RuntimeError("Could not encode annotated image.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc

    return {
        "filename": file.filename,
        "detection_count": len(_serialize_detections(result)),
        "detections": _serialize_detections(result),
        "annotated_image_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)) -> FileResponse:
    """Process an uploaded video and return an annotated MP4 for download."""
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_path, output_path = input_file.name, output_file.name
    input_file.close()
    output_file.close()

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded video is empty.")
        Path(input_path).write_bytes(content)

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise HTTPException(status_code=400, detail="Video could not be opened.")

        writer = None
        try:
            model = get_model()
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not 0 < fps <= 120:
                fps = 25.0

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    source=frame, conf=0.25, imgsz=512, verbose=False
                )[0]
                annotated = result.plot()
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        fps, (width, height)
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not initialize MP4 writer.")
                writer.write(annotated)
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if not Path(output_path).stat().st_size:
            raise HTTPException(status_code=400, detail="No decodable video frames found.")

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="inspection_result.mp4",
            background=lambda: (Path(input_path).unlink(missing_ok=True),
                                Path(output_path).unlink(missing_ok=True)),
        )
    except HTTPException:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        raise
    except Exception as exc:
        Path(input_path).unlink(missing_ok=True)
        Path(output_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}") from exc
