from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PoseEstimator:
    def __init__(self, model_path="yolov8s-pose.pt"):
        self.model = YOLO(model_path)
        self.results = None
        self.keypoint_labels = [
            "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
            "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
            "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
            "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
        ]
        self.results_dict = None

    def predict(self, image_path):
        self.image_path = image_path
        self.results = self.model.predict(image_path, verbose=False)

    def show_image(self):
        for r in self.results:
            im_array = r.plot()
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(im_rgb)
            plt.title("Pose Estimation with Keypoint Classes")
            plt.axis("off")
            plt.show()

    def keypoints_to_dict(self):
        best_person_dict = None
        best_conf_sum = -1

        for r in self.results:
            kpts_xy = r.keypoints.xy.cpu().numpy()      # [N, 17, 2]
            kpts_conf = r.keypoints.conf.cpu().numpy()  # [N, 17]
            boxes = r.boxes.xyxy.cpu().numpy()          # [N, 4]

            N = kpts_xy.shape[0]
            for i in range(N):
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1
                person_dict = {}
                total_conf = 0

                for j, label in enumerate(self.keypoint_labels):
                    key = label.lower().replace(" ", "_")
                    x, y = kpts_xy[i, j]
                    conf = kpts_conf[i, j]
                    total_conf += conf
                    x_norm = (x - x1) / w
                    y_norm = (y - y1) / h
                    person_dict[f"{key}_x"] = float(x_norm)
                    person_dict[f"{key}_y"] = float(y_norm)
                    person_dict[f"{key}_conf"] = float(conf)

                if total_conf > best_conf_sum:
                    best_conf_sum = total_conf
                    best_person_dict = person_dict

        self.results_dict = best_person_dict
        return best_person_dict

    def predict_video(self, video_path, save_path=None, display=True):
        """
        Process each frame in a video and extract pose keypoints.
        Optionally save or display annotated output.
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        keypoints_list = []

        if not cap.isOpened():
            print("❌ Error opening video file")
            return

        # Prepare writer if saving output
        out = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run pose prediction
            results = self.model.predict(frame, verbose=False)
            self.results = results
            keypoints = self.keypoints_to_dict()
            if keypoints:
                keypoints["frame"] = frame_idx
                keypoints_list.append(keypoints)

            # # Draw keypoints and skeleton
            # annotated_frame = results[0].plot()

            # if display:
            #     cv2.imshow("Pose Estimation", annotated_frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break

            # if out:
            #     out.write(annotated_frame)

            frame_idx += 1

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        df = pd.DataFrame(keypoints_list)
        print(f"✅ Processed {frame_idx} frames, extracted {len(df)} keypoint entries")
        return df
