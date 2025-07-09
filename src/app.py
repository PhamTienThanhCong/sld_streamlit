import streamlit as st
import cv2
import numpy as np
from collections import deque
from deploy_model import extract_frame_features, model, label_encoder, THRESHOLD
from PIL import Image
from io import BytesIO
import base64
import time

# --- UI setup ---
st.set_page_config(layout="wide")
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stMainBlockContainer {padding: 0;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Session state ---
if "start_camera" not in st.session_state:
    st.session_state.start_camera = False
if "labels" not in st.session_state:
    st.session_state.labels = []
if "window" not in st.session_state:
    st.session_state.window = deque(maxlen=30)

# --- Giao diện khởi động ---
if not st.session_state.start_camera:
    if st.button("▶️ Bắt đầu nhận diện"):
        st.session_state.start_camera = True
        st.rerun()

# --- Khi đã bật camera ---
if st.session_state.start_camera:
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Xoá kết quả"):
            st.session_state.labels = []
            st.session_state.window.clear()
    result_placeholder = st.empty()
    result_placeholder.success(f"Nhận diện: {' '.join(st.session_state.labels)}")

    # --- Giao diện webcam browser ---
    st.markdown("### 👇 Cho phép truy cập webcam và bấm chụp để nhận diện")
    html_code = """
    <div style="text-align: center">
        <video id="video" width="640" height="480" autoplay playsinline></video>
        <br />
        <button id="capture" style="margin-top: 10px;">📸 Chụp ảnh</button>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const capture = document.getElementById('capture');

        navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
            video.srcObject = stream;
        });

        capture.onclick = () => {
            canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg');
            const input = window.parent.document.querySelector('input[data-testid="stTextInput"]');
            if (input) {
                input.value = dataUrl;
                input.dispatchEvent(new Event('input', { bubbles: true }));
            }
        };
        </script>
    """
    st.components.v1.html(html_code, height=550)

    # --- Nhận ảnh base64 gửi từ JS ---
    img_data_url = st.text_input("Ảnh base64 từ webcam (ẩn)")

    if img_data_url.startswith("data:image"):
        img_bytes = base64.b64decode(img_data_url.split(",")[1])
        img = Image.open(BytesIO(img_bytes))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # --- Biến trạng thái ---
        if "prev_right" not in st.session_state:
            st.session_state.prev_right = st.session_state.prev_left = None
            st.session_state.prev_right_center = st.session_state.prev_left_center = None
            st.session_state.prev_right_shoulder_dists = (0.0, 0.0)
            st.session_state.prev_left_shoulder_dists = (0.0, 0.0)
            st.session_state.prev_shoulder_left = st.session_state.prev_shoulder_right = None

        # --- Extract features ---
        (
            frame_features,
            hand_detected,
            st.session_state.prev_right,
            st.session_state.prev_left,
            st.session_state.prev_right_center,
            st.session_state.prev_left_center,
            right_shoulder_dists,
            left_shoulder_dists,
            st.session_state.prev_shoulder_left,
            st.session_state.prev_shoulder_right,
        ) = extract_frame_features(
            frame,
            st.session_state.prev_right,
            st.session_state.prev_left,
            st.session_state.prev_right_center,
            st.session_state.prev_left_center,
            st.session_state.prev_right_shoulder_dists,
            st.session_state.prev_left_shoulder_dists,
            st.session_state.prev_shoulder_left,
            st.session_state.prev_shoulder_right,
        )

        st.session_state.window.append(frame_features)

        # --- Dự đoán ---
        if len(st.session_state.window) == 30:
            pred = model.predict(np.array(st.session_state.window)[np.newaxis, ...], verbose=0)
            confidence = np.max(pred)
            if confidence >= THRESHOLD:
                pred_class = np.argmax(pred, axis=1)
                predicted_label = label_encoder.inverse_transform(pred_class)[0]
                st.session_state.labels.append(predicted_label)
                result_placeholder.success(f"Nhận diện: {' '.join(st.session_state.labels)}")
            st.session_state.window.clear()

        # Hiển thị ảnh chụp
        st.image(frame, channels="BGR", caption="Ảnh vừa chụp", use_container_width=True)
