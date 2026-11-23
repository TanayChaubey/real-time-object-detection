"""Streamlit client for the YOLO11n Visual Inspection API."""

import base64
import os

import requests
import streamlit as st

API_URL = os.getenv("DETECTION_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Visual Inspection", page_icon="🔎", layout="wide")
st.title("YOLO11n Visual Inspection")
st.caption("Upload an image or video to inspect detected objects.")

try:
    health = requests.get(f"{API_URL}/health", timeout=3)
    health.raise_for_status()
    st.success("Detection API is online.")
except requests.RequestException:
    st.error("Detection API is offline. Start it with: uvicorn src.api:app --reload")
    st.stop()

image_tab, video_tab = st.tabs(["Image inspection", "Video inspection"])

with image_tab:
    uploaded_image = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="image"
    )
    if uploaded_image is not None and st.button("Run image inspection"):
        with st.spinner("Running YOLO11n..."):
            try:
                response = requests.post(
                    f"{API_URL}/detect/image",
                    files={"file": (uploaded_image.name, uploaded_image.getvalue(),
                                    uploaded_image.type)},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
                annotated = base64.b64decode(result["annotated_image_base64"])
                left, right = st.columns(2)
                left.image(uploaded_image, caption="Original", use_container_width=True)
                right.image(annotated, caption="Detections", use_container_width=True)
                st.metric("Objects detected", result["detection_count"])
                st.dataframe(result["detections"], use_container_width=True)
            except requests.RequestException as exc:
                st.error(f"Image inspection failed: {exc}")
            except (KeyError, ValueError) as exc:
                st.error(f"Invalid API response: {exc}")

with video_tab:
    uploaded_video = st.file_uploader(
        "Choose a video", type=["mp4", "avi", "mov"], key="video"
    )
    if uploaded_video is not None and st.button("Run video inspection"):
        with st.spinner("Processing video. Longer videos may take time..."):
            try:
                response = requests.post(
                    f"{API_URL}/detect/video",
                    files={"file": (uploaded_video.name, uploaded_video.getvalue(),
                                    uploaded_video.type)},
                    timeout=900,
                )
                response.raise_for_status()
                st.video(response.content)
                st.download_button(
                    "Download annotated video",
                    data=response.content,
                    file_name="inspection_result.mp4",
                    mime="video/mp4",
                )
            except requests.RequestException as exc:
                st.error(f"Video inspection failed: {exc}")
