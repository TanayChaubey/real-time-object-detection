# Week 4 — Integration of Computer Vision Solutions and Cross-functional Testing

## Project context

**Prototype:** YOLO11n Visual Inspection System  
**Use case:** A small quality-inspection workflow in which an operator submits product images or videos and receives object detections plus annotated media.

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
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the API in terminal 1:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Start the interface in terminal 2:

```bash
streamlit run app.py
```

Run automated tests:

```bash
pytest -q
```

The API documentation is available at `http://127.0.0.1:8000/docs` while the service is running.

## Test plan and acceptance criteria

| ID | Test | Expected result | Evidence/status |
|---|---|---|---|
| T01 | GET health endpoint | HTTP 200 and status `ok` | Automated test included; execution pending |
| T02 | Upload valid image | HTTP 200, detections and annotated image returned | Automated test included; execution pending |
| T03 | Upload invalid image bytes | HTTP 400 | Automated test included; execution pending |
| T04 | Upload empty image | HTTP 400 | Automated test included; execution pending |
| T05 | Upload valid video | Annotated MP4 returned | Manual system test pending |
| T06 | Upload unsupported/corrupt video | Clear client error | Manual system test pending |
| T07 | Use Streamlit client | Results displayed and video downloadable | Manual system test pending |
| T08 | Webcam inference | Existing CLI webcam path remains available | Hardware-dependent manual test pending |

## Challenges and engineering considerations

1. **Separating UI and inference:** REST endpoints provide a clear boundary between the Streamlit client and model code.
2. **Model initialization cost:** The API caches one model instance per worker rather than loading weights per request.
3. **Media decoding:** Uploaded bytes are decoded and checked before inference; invalid image/video inputs return client errors.
4. **Video resource handling:** Capture and writer handles are released in cleanup paths; temporary files are removed after response handling.
5. **Runtime variability:** Inference speed depends on hardware, model checkpoint, resolution, video codec and input size. No end-to-end FPS claim is made without a local measurement.

## Results and limitations

The integration code and automated test cases are provided in this repository. **The tests and live prototype have not been executed in the target environment as part of this GitHub change.** Therefore, no passing-test count, video-processing result, webcam result or end-to-end latency is claimed here. Run the commands above and replace each pending status with observed evidence before final submission.

The prototype uses the pretrained YOLO11n model and reports object detections; it does not independently classify products as defective or non-defective. A production quality-control system would require a domain-specific labelled dataset, a defect taxonomy, acceptance thresholds, authentication, upload-size limits and deployment monitoring.

## Week 4 completion checklist

- [x] Select application context and define component boundaries.
- [x] Implement REST API for image and video inference.
- [x] Implement a basic operator-facing prototype.
- [x] Add automated API tests with a mocked detector.
- [ ] Execute automated tests and record actual results.
- [ ] Perform live image and video system tests.
- [ ] Capture screenshots and measured end-to-end performance.
