import streamlit as st
import cv2
import numpy as np
from collections import deque
from deploy_model import extract_frame_features, model, label_encoder, THRESHOLD
import time

# Cấu hình layout rộng
st.set_page_config(layout="wide")

# Ẩn footer, menu, header
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

# Giao diện khởi động
if not st.session_state.start_camera:
    if st.button("▶️ Bắt đầu nhận diện"):
        st.session_state.start_camera = True
        st.rerun()

# Khi đã bắt đầu nhận diện
if st.session_state.start_camera:
    # Thanh công cụ
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🗑️ Xoá kết quả"):
            st.session_state.labels = []
    # with col2:
    #     if st.button("🗣️ Chuyển sang ngôn ngữ nói"):
    #         full_text = " ".join(st.session_state.labels)

    result_placeholder = st.empty()
    result_placeholder.success(f"Nhận diện: {' '.join(st.session_state.labels)}")

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Lỗi: Không thể mở webcam")
        st.session_state.start_camera = False
    else:
        stframe = st.empty()
        window = deque(maxlen=30)

        # Biến trạng thái cho extract_frame_features
        prev_right = prev_left = prev_right_center = prev_left_center = None
        prev_right_shoulder_dists = (0.0, 0.0)
        prev_left_shoulder_dists = (0.0, 0.0)
        prev_shoulder_left = prev_shoulder_right = None

        # Vòng lặp mô phỏng thời gian thực
        for _ in range(
            500
        ):  # bạn có thể chỉnh thành vòng lặp vô hạn nếu dùng threading
            ret, frame = cap.read()
            if not ret:
                st.error("Không thể đọc webcam")
                break

            # Trích xuất đặc trưng
            (
                frame_features,
                hand_detected,
                prev_right,
                prev_left,
                prev_right_center,
                prev_left_center,
                right_shoulder_dists,
                left_shoulder_dists,
                prev_shoulder_left,
                prev_shoulder_right,
            ) = extract_frame_features(
                frame,
                prev_right,
                prev_left,
                prev_right_center,
                prev_left_center,
                prev_right_shoulder_dists,
                prev_left_shoulder_dists,
                prev_shoulder_left,
                prev_shoulder_right,
            )

            window.append(frame_features)

            if len(window) == 30:
                pred = model.predict(np.array(window)[np.newaxis, ...], verbose=0)
                confidence = np.max(pred)
                if confidence >= THRESHOLD:
                    pred_class = np.argmax(pred, axis=1)
                    predicted_label = label_encoder.inverse_transform(pred_class)[0]
                    st.session_state.labels.append(predicted_label)
                    # Cập nhật kết quả mỗi lần có từ mới
                    result_placeholder.success(
                        f"Nhận diện: {' '.join(st.session_state.labels)}"
                    )
                window.clear()

            # Hiển thị webcam
            stframe.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.05)

        cap.release()
        st.session_state.start_camera = False
