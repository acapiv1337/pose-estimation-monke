from ultralytics import YOLO
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

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

    def predict(self, image_path):
        self.image_path = image_path
        self.results = self.model.predict(image_path)

    def show_image(self):
        for r in self.results:
            # Draw skeleton & keypoints (BGR)
            im_array = r.plot()
            # Convert to RGB for matplotlib
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            plt.figure(figsize=(8, 6))
            plt.imshow(im_rgb)

            # Get keypoints
            kpts = r.keypoints.xy[0].cpu().numpy()  # shape: [17, 2]

            # Label each keypoint
            for i, (x, y) in enumerate(kpts):
                plt.text(
                    x + 1, y - 1,                  # smaller offset
                    self.keypoint_labels[i],        # keypoint class
                    fontsize=5, color='yellow',
                    backgroundcolor='black'
                )

            plt.title("Pose Estimation with Keypoint Classes")
            plt.axis("off")
            plt.show()

    def keypoints_to_dict(self):
        """
        Returns a list of dictionaries for each detected person:
        [
            {
                "Nose": (x_norm, y_norm, conf),
                "Left Eye": (x_norm, y_norm, conf),
                ...
            },
            ...
        ]
        Coordinates are normalized relative to the person bounding box.
        """
        persons = []
        for r in self.results:
            kpts_xy = r.keypoints.xy.cpu().numpy()      # [N,17,2]
            kpts_conf = r.keypoints.conf.cpu().numpy()  # [N,17]
            boxes = r.boxes.xyxy.cpu().numpy()          # [N,4]

            N = kpts_xy.shape[0]  # number of people detected

            for i in range(N):
                x1, y1, x2, y2 = boxes[i]
                w, h = x2 - x1, y2 - y1

                person_dict = {}
                for j, label in enumerate(self.keypoint_labels):
                    x, y = kpts_xy[i, j]
                    conf = kpts_conf[i, j]
                    # Normalize relative to bounding box
                    x_norm = (x - x1) / w
                    y_norm = (y - y1) / h
                    person_dict[label] = (float(x_norm), float(y_norm), float(conf))
                persons.append(person_dict)

        return persons
