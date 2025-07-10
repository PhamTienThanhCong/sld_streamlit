import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
import cv2
from collections import deque
from deploy_model import extract_frame_features, model, label_encoder, THRESHOLD
import time
import json

METADATA_PATH = './data/metadata.json'
metadata = {}
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# -------------------- UI Setup --------------------
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

if "labels" not in st.session_state:
    st.session_state.labels = []

if st.button("🗑️ Xoá kết quả"):
    st.session_state.labels.clear()

result_placeholder = st.empty()

result_placeholder.success(f"Nhận diện: ")

# -------------------- Video Processor --------------------
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        # Biến lưu kết quả mới nhận diện
        self.result = None
        self.window = deque(maxlen=30)

        # Biến trạng thái cho frame trước
        self.prev_right = self.prev_left = None
        self.prev_right_center = self.prev_left_center = None
        self.prev_right_shoulder_dists = (0.0, 0.0)
        self.prev_left_shoulder_dists = (0.0, 0.0)
        self.prev_shoulder_left = self.prev_shoulder_right = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Kiểm tra ảnh hợp lệ (tránh lỗi MediaPipe)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Trích xuất đặc trưng
        (
            frame_features,
            hand_detected,
            self.prev_right,
            self.prev_left,
            self.prev_right_center,
            self.prev_left_center,
            right_shoulder_dists,
            left_shoulder_dists,
            self.prev_shoulder_left,
            self.prev_shoulder_right,
        ) = extract_frame_features(
            img,
            self.prev_right,
            self.prev_left,
            self.prev_right_center,
            self.prev_left_center,
            self.prev_right_shoulder_dists,
            self.prev_left_shoulder_dists,
            self.prev_shoulder_left,
            self.prev_shoulder_right,
        )

        self.window.append(frame_features)

        if len(self.window) == 30:
            pred = model.predict(np.array(self.window)[np.newaxis, ...], verbose=0)
            confidence = np.max(pred)
            if confidence >= THRESHOLD:
                pred_class = np.argmax(pred, axis=1)
                predicted_label = label_encoder.inverse_transform(pred_class)[0]
                self.result = metadata.get(predicted_label, "Không nhận diện được")
            self.window.clear()

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -------------------- Giao tiếp với processor --------------------
ctx = webrtc_streamer(
    key="sign_lang",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    async_processing=True,
)

# Cập nhật giao diện từ kết quả xử lý video mỗi 1 giây
while ctx.state.playing and ctx.video_processor:
    if ctx.video_processor.result:
        st.session_state.labels.append(ctx.video_processor.result)
        result_placeholder.success(f"Nhận diện: {' / '.join(st.session_state.labels)}")
        ctx.video_processor.result = None  # reset để không lặp lại
    else:
        # Hiển thị kết quả hiện tại nếu có
        if st.session_state.labels:
            result_placeholder.success(f"Nhận diện: {' / '.join(st.session_state.labels)}")
    time.sleep(1)