"""
Extract pose keypoint features from all images in data/ using YOLOv8s-pose.
Saves features to CSV and numpy array files.
"""
import sys
import os
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

# --- Config ---
DATA_DIR = Path("data")
MODEL_PATH = "backend/yolov8s-pose.pt"
OUTPUT_DIR = Path("features")
CLASS_NAMES = ["heart-attack", "idea", "stand", "think"]

# 17 keypoint labels (COCO format from yolov8-pose)
KEYPOINT_LABELS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load model ---
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Extract features ---
all_rows = []
all_features = []
all_labels = []

for class_name in CLASS_NAMES:
    class_dir = DATA_DIR / class_name
    if not class_dir.exists():
        print(f"  ⚠ Skipping {class_name}: directory not found")
        continue

    image_paths = sorted(class_dir.glob("*"))
    print(f"\n📁 {class_name}: {len(image_paths)} images")

    for i, img_path in enumerate(image_paths):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue

        try:
            results = model.predict(str(img_path), verbose=False, device=device)
        except Exception as e:
            print(f"  ❌ Error on {img_path.name}: {e}")
            continue

        # Get the best person detection (highest total confidence)
        best_features = None
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
                feat = {}
                total_conf = 0

                for j, label in enumerate(KEYPOINT_LABELS):
                    x, y = kpts_xy[p, j]
                    conf = kpts_conf[p, j]
                    total_conf += float(conf)

                    # Normalize keypoints relative to bounding box
                    if w > 0 and h > 0:
                        feat[f"{label}_x"] = float((x - x1) / w)
                        feat[f"{label}_y"] = float((y - y1) / h)
                    else:
                        feat[f"{label}_x"] = 0.0
                        feat[f"{label}_y"] = 0.0
                    feat[f"{label}_conf"] = float(conf)

                # Also include raw keypoints and box dimensions
                for j, label in enumerate(KEYPOINT_LABELS):
                    feat[f"{label}_raw_x"] = float(kpts_xy[p, j, 0])
                    feat[f"{label}_raw_y"] = float(kpts_xy[p, j, 1])

                feat["bbox_w"] = float(w)
                feat["bbox_h"] = float(h)

                if total_conf > best_conf_sum:
                    best_conf_sum = total_conf
                    best_features = feat

        if best_features is not None:
            best_features["image"] = f"{class_name}/{img_path.name}"
            best_features["class"] = class_name
            best_features["class_id"] = CLASS_NAMES.index(class_name)
            best_features["total_conf"] = best_conf_sum
            all_rows.append(best_features)

        if (i + 1) % 20 == 0:
            print(f"    processed {i+1}/{len(image_paths)}", end="\r")

    print(f"    extracted features for {len([r for r in all_rows if r['class']==class_name])} images")

print(f"\n✅ Total features extracted: {len(all_rows)}")

if len(all_rows) == 0:
    print("❌ No features extracted. Check your model and images.")
    sys.exit(1)

# --- Save as CSV ---
fieldnames = list(all_rows[0].keys())
csv_path = OUTPUT_DIR / "pose_features.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)
print(f"📄 CSV saved: {csv_path} ({len(all_rows)} rows, {len(fieldnames)} columns)")

# --- Save as numpy for ML ---
# Feature vector: 51 features (17 keypoints × 3: x_norm, y_norm, conf) + bbox_w, bbox_h
feature_keys = []
for label in KEYPOINT_LABELS:
    feature_keys.extend([f"{label}_x", f"{label}_y", f"{label}_conf"])
feature_keys.extend(["bbox_w", "bbox_h"])

features_array = np.array([[row[k] for k in feature_keys] for row in all_rows], dtype=np.float32)
labels_array = np.array([row["class_id"] for row in all_rows], dtype=np.int32)
image_names = [row["image"] for row in all_rows]

np.save(OUTPUT_DIR / "features.npy", features_array)
np.save(OUTPUT_DIR / "labels.npy", labels_array)
with open(OUTPUT_DIR / "image_names.txt", "w") as f:
    f.write("\n".join(image_names))
# Save feature names
with open(OUTPUT_DIR / "feature_names.txt", "w") as f:
    f.write("\n".join(feature_keys))
# Save class names
with open(OUTPUT_DIR / "class_names.txt", "w") as f:
    f.write("\n".join(CLASS_NAMES))

print(f"🔢 Features array: {features_array.shape} — {len(all_rows)} samples, {len(feature_keys)} features each")
print(f"🏷️  Labels array: {labels_array.shape}")

# Class distribution
print(f"\n📊 Class distribution:")
for cls in CLASS_NAMES:
    count = sum(1 for r in all_rows if r["class"] == cls)
    print(f"  {cls}: {count}")
