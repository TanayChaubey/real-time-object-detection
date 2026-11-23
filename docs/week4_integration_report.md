# Week 4 — Integration of Computer Vision Solutions and Cross-functional Testing

## Project context

**Prototype:** YOLO11n Visual Inspection System  
**Use case:** A simulated visual-inspection workflow in which an operator submits product images or videos and receives object detections plus annotated media.

This is a general object-detection prototype, not a validated defect-detection system.

## Integration strategy

The existing YOLO11n inference module is exposed through a FastAPI REST service. A Streamlit application acts as the operator-facing client. This separates the computer-vision inference layer from the user interface and allows another client to consume the same HTTP endpoints.

### Architecture

```text
Operator
   |
   v
Streamlit interface (app.py)
   | HTTP multipart upload
   v
FastAPI service (src/api.py)
   | decode / validate input
   v
YOLO11n model (Ultralytics)
   | detections + annotated media
   v
JSON response / MP4 response
   |
   v
Streamlit results and download
```

### API contract

| Method | Endpoint | Purpose | Successful response |
|---|---|---|---|
| GET | `/health` | Check service availability | JSON status |
| POST | `/detect/image` | Detect objects in uploaded image | JSON detections and base64 annotated image |
| POST | `/detect/video` | Process uploaded video | Annotated MP4 file |

The model is loaded lazily and reused within the API process to avoid loading weights for every request. The default checkpoint is `yolo11n.pt`; set `YOLO_MODEL` to select another compatible checkpoint.

## Implementation

- `src/api.py`: REST endpoints, input decoding, inference, response serialization and temporary video handling.
- `app.py`: Streamlit client for image/video upload, result display and annotated video download.
- `tests/test_api.py`: API tests using a fake detector so tests do not require model downloads or GPU hardware.

## Setup and execution

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API in terminal 1:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Start the interface in terminal 2:

```bash
streamlit run app.py
```

Run automated tests from the repository root:

```bash
python -m pytest -q
```

Use `python -m pytest` so the current project directory is included on Python's import path. In the tested Anaconda environment, invoking `pytest -q` directly initially failed during collection with `ModuleNotFoundError: No module named 'src'`; the module invocation resolved this.

The API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.

## Test plan and observed results

Manual tests were performed locally on a MacBook Air M1 in the Anaconda environment. Automated tests use a mocked detector; manual image/video checks exercise the real pretrained YOLO11n checkpoint.

| ID | Test | Expected result | Observed result |
|---|---|---|---|
| T01 | GET API documentation / OpenAPI schema | API starts and documentation is served | **Passed** — `/docs` and `/openapi.json` both returned HTTP 200. |
| T02 | Upload valid image | HTTP 200, detections and annotated image returned | **Passed** — a test image returned two detections: `dog` (confidence 0.92261) and `frisbee` (confidence 0.44411), with bounding-box coordinates and an annotated-image payload. |
| T03 | Upload invalid image bytes | HTTP 400 | **Automated test passed** — covered by the test suite. |
| T04 | Upload empty image | HTTP 400 | **Automated test passed** — covered by the test suite. |
| T05 | Upload valid video | Annotated MP4 returned and playable | **Passed** — endpoint returned `inspection_result.mp4` with `video/mp4` content type and content length 6,387,476 bytes. The downloaded video played and displayed a `person` detection with a blue bounding box and confidence around 0.95. |
| T06 | Upload unsupported/corrupt video | Clear client error | **Not tested** — requires a separate negative test. |
| T07 | Use Streamlit client | Results displayed to the operator | **Partially verified** — image upload through the frontend displayed two detections. Record a separate frontend video-playback result if tested through Streamlit. |
| T08 | Webcam inference | Existing CLI webcam path remains available | **Not tested** — hardware-dependent. |

Automated test command and result:

```text
python -m pytest -q
....                                                                     [100%]
4 passed in 1.08s
```

The direct `pytest -q` command initially failed during test collection because the `src` package was not resolved on the import path. Running the suite as `python -m pytest -q` from the repository root succeeded.

## Challenges and engineering considerations

1. **Separating UI and inference:** REST endpoints provide a clear boundary between the Streamlit client and model code.
2. **Model initialization cost:** The API caches one model instance per worker rather than loading weights per request.
3. **Media decoding:** Uploaded bytes are decoded and checked before inference; invalid image inputs return client errors as covered by the automated tests.
4. **Video resource handling:** Capture and writer handles are released in cleanup paths; temporary files are removed after response handling.
5. **Runtime variability:** Inference speed depends on hardware, model checkpoint, resolution, video codec and input size. No end-to-end FPS or latency claim is made because these were not measured in this integration test.

## Results and limitations

The local integration was exercised successfully for API startup, image inference, annotated image output, video inference and annotated MP4 playback. The automated API test suite passed all four tests when run with `python -m pytest -q`.

The prototype uses the pretrained YOLO11n model and reports general object detections. It does **not** independently classify products as defective or non-defective. The dog/frisbee and person detections demonstrate general COCO-style object recognition, not defect recognition or quality-control accuracy.

The current verification does not establish performance under concurrent load, security against untrusted uploads, behaviour on corrupt video files, webcam operation, or production readiness. A production quality-control system would require a domain-specific labelled dataset, a defect taxonomy, validated acceptance thresholds, authentication, upload-size limits, deployment monitoring and measured end-to-end performance.

## Week 4 completion checklist

- [x] Select application context and define component boundaries.
- [x] Implement REST API for image and video inference.
- [x] Implement a basic operator-facing prototype.
- [x] Add automated API tests with a mocked detector.
- [x] Execute automated tests and record actual results (4 passed).
- [x] Perform live image and video inference checks.
- [ ] Test unsupported/corrupt video handling.
- [ ] Test webcam path, if required and available.
- [ ] Capture and attach screenshots as submission evidence.
- [ ] Measure end-to-end performance if performance figures are required.
