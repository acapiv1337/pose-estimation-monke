from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2, base64, numpy as np
from ultralytics import YOLO
import xgboost as xgb
import io
from PIL import Image
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# --- Load models ---
pose_model = YOLO(str(Path(__file__).parent / "yolov8s-pose.pt"))
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(Path(__file__).parent / "xgb_model.json"))

CLASS_NAMES = ["heart-attack", "idea", "stand", "think"]

# 17 COCO keypoint labels (same order as features)
KEYPOINT_LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

processing = False  # global flag to skip frames


def extract_features(results):
    """Extract the 53-feature vector from YOLO pose results (best person)."""
    best_feats = None
    best_conf_sum = -1

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        kpts_xy = r.keypoints.xy.cpu().numpy()      # [N, 17, 2]
        kpts_conf = r.keypoints.conf.cpu().numpy()   # [N, 17]
        boxes = r.boxes.xyxy.cpu().numpy()           # [N, 4]

        for p in range(len(kpts_xy)):
            x1, y1, x2, y2 = boxes[p]
            w, h = x2 - x1, y2 - y1
            feats = []
            total_conf = 0

            for j in range(17):
                x, y = kpts_xy[p, j]
                conf = kpts_conf[p, j]
                total_conf += float(conf)
                if w > 0 and h > 0:
                    feats.extend([float((x - x1) / w), float((y - y1) / h), float(conf)])
                else:
                    feats.extend([0.0, 0.0, float(conf)])

            feats.extend([float(w), float(h)])

            if total_conf > best_conf_sum:
                best_conf_sum = total_conf
                best_feats = np.array([feats], dtype=np.float32)

    return best_feats


@app.websocket("/ws/pose")
async def websocket_pose(websocket: WebSocket):
    global processing
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if processing:
                continue
            processing = True

            # Decode base64 image from frontend
            img_data = data["image"].split(",")[1]
            img_bytes = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            frame = np.array(img)

            # Run pose prediction (internal — for feature extraction only)
            results = pose_model.predict(frame, verbose=False)

            # Run XGBoost classification
            feats = extract_features(results)
            predicted_class = None
            confidence = 0.0
            if feats is not None:
                proba = xgb_model.predict_proba(feats)[0]
                pred_id = int(np.argmax(proba))
                confidence = float(proba[pred_id])
                predicted_class = CLASS_NAMES[pred_id]

            # Encode original frame (no overlay, no skeleton, no label)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", frame_bgr)
            encoded = base64.b64encode(buffer).decode("utf-8")

            response = {"image": encoded}
            if predicted_class:
                response["class"] = predicted_class
                response["confidence"] = confidence

            await websocket.send_json(response)
            processing = False
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        processing = False
        await websocket.close()
