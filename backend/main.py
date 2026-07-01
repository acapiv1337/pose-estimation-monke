from fastapi import FastAPI, WebSocket, UploadFile, File
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

processing = False  # global flag to skip frames


def extract_features(results):
    """Extract 22-feature vector from YOLO pose results (best person).
    Features: (x,y) of 11 upper-body keypoints: nose, eyes, ears, shoulders, elbows, wrists.
    Values are normalized relative to bounding box.
    """
    best_feats = None
    best_conf_sum = -1

    # Only the first 11 COCO keypoints (upper body)
    KEYPOINT_INDICES = list(range(11))  # nose=0 .. right_wrist=10

    for r in results:
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            continue

        kpts_xy = r.keypoints.xy.cpu().numpy()
        kpts_conf = r.keypoints.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for p in range(len(kpts_xy)):
            x1, y1, x2, y2 = boxes[p]
            w, h = x2 - x1, y2 - y1
            feats = []
            total_conf = 0

            for j in KEYPOINT_INDICES:
                x, y = kpts_xy[p, j]
                conf = kpts_conf[p, j]
                total_conf += float(conf)
                if w > 0 and h > 0:
                    feats.extend([float((x - x1) / w), float((y - y1) / h)])
                else:
                    feats.extend([0.0, 0.0])

            if total_conf > best_conf_sum:
                best_conf_sum = total_conf
                best_feats = np.array([feats], dtype=np.float32)

    return best_feats


def predict_image(frame: np.ndarray):
    """Run YOLO + XGBoost on a single frame and return prediction."""
    results = pose_model.predict(frame, verbose=False)
    feats = extract_features(results)
    if feats is None:
        return None, 0.0
    proba = xgb_model.predict_proba(feats)[0]
    pred_id = int(np.argmax(proba))
    return CLASS_NAMES[pred_id], float(proba[pred_id])


@app.post("/predict")
async def upload_predict(file: UploadFile = File(...)):
    """Upload an image and get pose classification."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    frame = np.array(img)
    predicted_class, confidence = predict_image(frame)
    if predicted_class:
        return {"class": predicted_class, "confidence": confidence}
    return {"class": None, "confidence": 0.0, "error": "No person detected"}


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

            # Run pose prediction (for feature extraction only)
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

            # Encode original frame (no overlay)
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
