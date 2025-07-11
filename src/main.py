from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import base64
import numpy as np
import cv2
import json
from deploy_model import extract_frame_features, model, label_encoder, THRESHOLD
from collections import deque
from pyngrok import ngrok
import uvicorn

# Load metadata
METADATA_PATH = './data/metadata.json'
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

# FastAPI instance
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the SLD Streamlit API!"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    prev_state = {
        "prev_right": None,
        "prev_left": None,
        "prev_right_center": None,
        "prev_left_center": None,
        "prev_right_shoulder_dists": (0.0, 0.0),
        "prev_left_shoulder_dists": (0.0, 0.0),
        "prev_shoulder_left": None,
        "prev_shoulder_right": None,
    }
    window = deque(maxlen=30)

    while True:
        try:
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            features, hand_detected, *states = extract_frame_features(
                frame,
                prev_state["prev_right"],
                prev_state["prev_left"],
                prev_state["prev_right_center"],
                prev_state["prev_left_center"],
                prev_state["prev_right_shoulder_dists"],
                prev_state["prev_left_shoulder_dists"],
                prev_state["prev_shoulder_left"],
                prev_state["prev_shoulder_right"],
            )

            (
                prev_state["prev_right"],
                prev_state["prev_left"],
                prev_state["prev_right_center"],
                prev_state["prev_left_center"],
                prev_state["prev_right_shoulder_dists"],
                prev_state["prev_left_shoulder_dists"],
                prev_state["prev_shoulder_left"],
                prev_state["prev_shoulder_right"],
            ) = states

            window.append(features)
            if len(window) == 30:
                pred = model.predict(np.array([window]), verbose=0)
                confidence = float(np.max(pred))
                if confidence >= THRESHOLD:
                    pred_class = np.argmax(pred, axis=1)
                    label = label_encoder.inverse_transform(pred_class)[0]
                    await websocket.send_json({"result": metadata[label], "confidence": confidence})
                else:
                    await websocket.send_json({"result": None})
                window.clear()

        except Exception as e:
            print("WebSocket error:", e)
            break

if __name__ == "__main__":
    # ✅ Khởi tạo tunnel ngrok
    public_url = ngrok.connect(8000, bind_tls=True)
    print(f"🌐 Public Ngrok URL: {public_url}")
    
    # ✅ Chạy FastAPI
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
