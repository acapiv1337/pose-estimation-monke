"""
WebRTC-based pose classification server.
Receives H.264/VP8 video stream via WebRTC, runs YOLO pose + XGBoost classification,
sends results back via DataChannel.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from ultralytics import YOLO
import xgboost as xgb
from aiortc import RTCPeerConnection, RTCSessionDescription
import asyncio
import json
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

# 17 COCO keypoint labels
KEYPOINT_LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

pcs = set()


def extract_features(results):
    """Extract the 53-feature vector from YOLO pose results (best person)."""
    best_feats = None
    best_conf_sum = -1

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


async def process_video(track, channel, pc):
    """Process incoming video frames: YOLO pose + XGBoost classification."""
    loop = asyncio.get_event_loop()
    frame_count = 0
    INFERENCE_INTERVAL = 3  # run inference every N frames (~10 FPS at 30 FPS input)

    while True:
        try:
            frame = await asyncio.wait_for(track.recv(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            break

        frame_count += 1
        if frame_count % INFERENCE_INTERVAL != 0:
            continue

        # aiortc/av gives BGR24 directly — OpenCV-compatible
        img = frame.to_ndarray(format="bgr24")

        # Run YOLO pose in thread pool (blocking call)
        results = await loop.run_in_executor(
            None, lambda: pose_model.predict(img, verbose=False)
        )

        # Extract features and classify
        feats = extract_features(results)
        if feats is not None:
            proba = xgb_model.predict_proba(feats[np.newaxis, ...])[0]
            pred_id = int(np.argmax(proba))
            conf = float(proba[pred_id])
            cls = CLASS_NAMES[pred_id]

            # Send result via DataChannel
            if channel.readyState == "open":
                try:
                    channel.send(json.dumps({
                        "class": cls,
                        "confidence": round(conf, 3),
                        "class_id": pred_id,
                    }))
                except Exception:
                    pass

    # Cleanup
    pcs.discard(pc)
    try:
        await pc.close()
    except Exception:
        pass


@app.post("/offer")
async def offer(request: Request):
    """Receive WebRTC SDP offer from browser, return answer."""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Create DataChannel for sending classification results
    channel = pc.createDataChannel("classification")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState in ["failed", "closed"]:
            pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            asyncio.ensure_future(process_video(track, channel, pc))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }


@app.on_event("shutdown")
async def shutdown():
    """Clean up all peer connections on shutdown."""
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
