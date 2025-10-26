from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import cv2, base64, numpy as np
from ultralytics import YOLO
import io
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pose_model = YOLO("yolov8s-pose.pt")
processing = False  # global flag to skip frames

@app.websocket("/ws/pose")
async def websocket_pose(websocket: WebSocket):
    global processing
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if processing:
                continue  # skip if still processing previous frame
            processing = True

            # Decode base64 image from frontend
            img_data = data["image"].split(",")[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame = np.array(img)

            # Run pose prediction
            results = pose_model.predict(frame, verbose=False)
            annotated = results[0].plot()

            # Encode back to base64
            _, buffer = cv2.imencode(".jpg", annotated)
            encoded = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_json({"image": encoded})

            processing = False
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        await websocket.close()
