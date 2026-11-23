"""API tests use a fake detector; no model download or GPU is required."""

import io

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src import api


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[1, 2, 20, 30]])
        self.conf = FakeTensor([0.91])
        self.cls = FakeTensor([0])


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "object"}

    def plot(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)


class FakeModel:
    def predict(self, **kwargs):
        return [FakeResult()]


client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_image_endpoint_returns_detections(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    response = client.post(
        "/detect/image",
        files={"file": ("sample.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["detection_count"] == 1
    assert payload["detections"][0]["label"] == "object"
    assert payload["detections"][0]["confidence"] == 0.91
    assert payload["annotated_image_base64"]


def test_image_endpoint_rejects_invalid_file():
    response = client.post(
        "/detect/image",
        files={"file": ("not-image.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


def test_image_endpoint_rejects_empty_file():
    response = client.post(
        "/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
