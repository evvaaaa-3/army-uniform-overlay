from pathlib import Path

import cv2
from ultralytics import YOLO


class PoseDetector:
    """
    YOLOv8 pose detector.

    Output format stays compatible with the rest of our modular code:
    pose["left_shoulder"] = {"x": ..., "y": ..., "vis": ...}
    """

    def __init__(self, model_path="yolov8n-pose.pt", confidence_threshold=0.35):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO pose model not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold

        # COCO keypoint order used by YOLOv8-pose
        self.keypoint_names = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

    def detect(self, frame_bgr):
        results = self.model.predict(
            frame_bgr,
            verbose=False,
            conf=self.confidence_threshold,
        )

        if not results:
            return None, None

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None, result

        # Pick the largest detected person box
        if result.boxes is None or len(result.boxes) == 0:
            person_index = 0
        else:
            boxes = result.boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            person_index = int(areas.argmax())

        xy = result.keypoints.xy[person_index].cpu().numpy()

        if result.keypoints.conf is not None:
            conf = result.keypoints.conf[person_index].cpu().numpy()
        else:
            conf = [1.0] * len(self.keypoint_names)

        pose = {}

        for i, name in enumerate(self.keypoint_names):
            x, y = xy[i]
            v = float(conf[i])

            pose[name] = {
                "x": int(x),
                "y": int(y),
                "vis": v,
            }

        return pose, result

    def draw(self, frame_bgr, results):
        """
        Minimal drawing method for compatibility.
        YOLO result plotting is heavier, so for now we draw keypoints only.
        """
        pose, _ = self.detect(frame_bgr)

        if pose is None:
            return frame_bgr

        for _, p in pose.items():
            if p["vis"] > self.confidence_threshold:
                cv2.circle(frame_bgr, (p["x"], p["y"]), 4, (0, 255, 255), -1)

        return frame_bgr