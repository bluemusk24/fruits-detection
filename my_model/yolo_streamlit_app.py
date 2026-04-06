import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# -----------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# BBOX COLORS  (Tableau-10)
# -----------------------------------------------------------------------
BBOX_COLORS = [
    (164, 120,  87), ( 68, 148, 228), ( 93,  97, 209), (178, 182, 133),
    ( 88, 159, 106), ( 96, 202, 231), (159, 124, 168), (169, 162, 241),
    ( 98, 118, 150), (172, 176, 184),
]


# -----------------------------------------------------------------------
# MODEL LOADER
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_yolo_model(model_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name
    return YOLO(tmp_path, task="detect")


# -----------------------------------------------------------------------
# ANNOTATION
# -----------------------------------------------------------------------
def annotate_frame(frame, results, conf_thresh: float):
    labels_map = results[0].names
    detections = results[0].boxes
    found = []

    for det in detections:
        conf = det.conf.item()
        if conf < conf_thresh:
            continue

        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        class_id = int(det.cls.item())
        name = labels_map[class_id]
        color = BBOX_COLORS[class_id % len(BBOX_COLORS)]

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        label = f"{name}: {int(conf * 100)}%"
        (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(ymin, lh + 10)
        cv2.rectangle(frame,
                      (xmin, y_label - lh - 10),
                      (xmin + lw, y_label + base - 10),
                      color, cv2.FILLED)
        cv2.putText(frame, label,
                    (xmin, y_label - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        found.append((name, conf))

    return frame, found


# -----------------------------------------------------------------------
# SIDEBAR — model upload + settings
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("YOLO Object Detector")
    st.divider()

    st.subheader("Model")
    model_file = st.file_uploader(
        "Upload YOLO weights (.pt)",
        type=["pt"],
        help="Ultralytics YOLOv8/v5 .pt weights file",
    )

    st.subheader("Settings")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.40, 0.05)

    model = None
    if model_file:
        with st.spinner("Loading model…"):
            try:
                model = load_yolo_model(model_file.read())
                st.success(f"Model loaded — {len(model.names)} classes")
                with st.expander("Class list"):
                    for idx, name in model.names.items():
                        st.text(f"{idx}: {name}")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
    else:
        st.info("Upload a .pt file to begin")


# -----------------------------------------------------------------------
# MAIN — source selector
# -----------------------------------------------------------------------
col_src, col_view = st.columns([1, 2], gap="large")

with col_src:
    st.subheader("Input Source")
    source_type = st.radio(
        "Choose source type",
        ["Image", "Video", "Webcam snapshot"],
        label_visibility="collapsed",
    )

with col_view:
    st.subheader("Detection Output")


# -----------------------------------------------------------------------
# IMAGE
# -----------------------------------------------------------------------
if source_type == "Image":
    with col_src:
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )

    if uploaded and model:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model(frame, verbose=False)
        ann, found = annotate_frame(frame.copy(), results, conf_thresh)

        with col_view:
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.metric("Objects detected", len(found))
            if found:
                for name, conf in found:
                    st.text(f"• {name}  {conf*100:.0f}%")

    elif uploaded and not model:
        with col_view:
            st.warning("Upload a model in the sidebar first.")


# -----------------------------------------------------------------------
# VIDEO
# -----------------------------------------------------------------------
elif source_type == "Video":
    with col_src:
        uploaded = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv", "wmv"],
        )
        run_btn = st.button("▶ Run inference", disabled=(uploaded is None or model is None))
        stop_btn = st.button("⏹ Stop")

    if run_btn and uploaded and model:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps_buffer = []
        frame_idx = 0

        with col_view:
            stframe = st.empty()
            progress_bar = st.progress(0, text="Processing…")

        while cap.isOpened():
            if stop_btn:
                break
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, verbose=False)
            ann, found = annotate_frame(frame.copy(), results, conf_thresh)
            elapsed = time.perf_counter() - t0
            fps_buffer.append(1.0 / max(elapsed, 1e-6))
            avg_fps = np.mean(fps_buffer[-60:])

            cv2.putText(ann, f"FPS: {avg_fps:.1f}",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 160), 2)
            cv2.putText(ann, f"Objects: {len(found)}",
                        (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 229, 160), 2)

            stframe.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

            frame_idx += 1
            pct = min(int(frame_idx / total_frames * 100), 100)
            progress_bar.progress(pct, text=f"Frame {frame_idx} · {avg_fps:.1f} FPS")

        cap.release()
        progress_bar.progress(100, text="Done ✓")

    elif uploaded and not model:
        with col_view:
            st.warning("Upload a model in the sidebar first.")


# -----------------------------------------------------------------------
# WEBCAM SNAPSHOT
# -----------------------------------------------------------------------
elif source_type == "Webcam snapshot":
    with col_src:
        cam_input = st.camera_input("Capture a frame")

    if cam_input and model:
        file_bytes = np.frombuffer(cam_input.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model(frame, verbose=False)
        ann, found = annotate_frame(frame.copy(), results, conf_thresh)

        with col_view:
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.metric("Objects detected", len(found))
            if found:
                for name, conf in found:
                    st.text(f"• {name}  {conf*100:.0f}%")

    elif cam_input and not model:
        with col_view:
            st.warning("Upload a model in the sidebar first.")


# -----------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------
st.divider()
st.caption("YOLO Detector · Ultralytics · Streamlit")
