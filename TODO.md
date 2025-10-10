# 🧠 TODO List — Real-Time Pose Estimation & Classification

## 🎯 Objective
Develop a real-time webcam application (Dockerized) that uses **pose estimation** to detect and extract body keypoints (e.g. finger, head, shoulder), then classify the pose to determine which image or action it represents.

---

## 🏗️ Phase 1 — Setup & Environment
- [ ] Initialize project structure
  - [ ] `/pose_estimation/` for model logic
  - [ ] `/api/` for REST or FastAPI interface
  - [ ] `/docker/` for Docker-related files
- [ ] Create `Dockerfile` for the main app
- [ ] Add `docker-compose.yml` for services (e.g., backend + webcam stream)
- [ ] Prepare `requirements.txt` (OpenCV, Ultralytics, FastAPI, etc.)
- [ ] Verify webcam access in Docker container

---

## 🧍 Phase 2 — Pose Estimation
- [ ] Choose model: Ultralytics YOLOv8 Pose
- [ ] Load pretrained model
- [ ] Capture webcam frames in real-time
- [ ] Extract keypoints (Nose, Eyes, Ears, Shoulders, Elbows, Wrists, Hips, Knees, Ankles, Fingers)
- [ ] Visualize keypoints overlay on webcam feed
- [ ] Normalize coordinates for model input

---

## 🧩 Phase 3 — Keypoint Endpoint Extraction
- [ ] Define JSON output structure for keypoints:
  ```json
  {
    "nose": [x, y],
    "left_wrist": [x, y],
    "right_wrist": [x, y],
    ...
  }

## 📝 Phase 4 — Classification
- [ ] Classify poses using a pre-trained model
  - Use the `keypoints` output from Phase 3 to train a classifier
  - Use the classifier to classify poses
  - Predict the pose class (e.g., "attack", "idea", "stand", "think")
  
## 📈 Phase 5 — Production / Web App
- [ ] Interface with live time feed webcam to detect poses