from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.geometry import distance, midpoint, to_np


@dataclass
class PoseValidationResult:
    valid: bool
    message: str
    error_code: Optional[str] = None
    details: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    anchors: Optional[dict] = None

    @property
    def ok(self):
        return self.valid

    def to_dict(self):
        return {
            "valid": self.valid,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "metrics": self.metrics,
        }


class FrontPoseValidator:
    """
    Mode-aware validation for the still-photo MVP.

    YOLO pose cannot truly prove front vs back, so "front" means face/shoulder
    geometry is plausible for a mostly upright person facing the camera.
    """

    def __init__(self, visibility_threshold=0.40):
        self.visibility_threshold = visibility_threshold

    def _visible(self, pose, name, threshold=None):
        threshold = self.visibility_threshold if threshold is None else threshold
        return pose is not None and name in pose and pose[name]["vis"] >= threshold

    def _point(self, pose, name):
        return to_np(pose[name])

    def _fail(self, message, error_code, details=None, metrics=None):
        return PoseValidationResult(
            False,
            message,
            error_code=error_code,
            details=details or [],
            metrics=metrics or {},
        )

    def _visible_points(self, pose, threshold=0.25):
        if pose is None:
            return {}
        return {
            name: to_np(point)
            for name, point in pose.items()
            if point.get("vis", 0.0) >= threshold
        }

    def validate(self, pose, image_shape, mode="upper_body"):
        height, width = image_shape[:2]
        mode = mode if mode in {"upper_body", "full_body"} else "upper_body"

        if pose is None:
            return self._fail(
                "No person detected. Please face the camera and retake the photo.",
                "NO_PERSON_DETECTED",
                ["YOLO pose did not return a person"],
            )

        visible_points = self._visible_points(pose)
        if not visible_points:
            return self._fail(
                "No person detected. Please face the camera and retake the photo.",
                "NO_PERSON_DETECTED",
                ["No visible pose landmarks detected"],
            )

        if not self._visible(pose, "left_shoulder") or not self._visible(pose, "right_shoulder"):
            return self._fail(
                "Keep both shoulders visible and face the camera.",
                "SHOULDERS_NOT_VISIBLE",
                ["No clear shoulder landmarks detected"],
            )

        left_shoulder = self._point(pose, "left_shoulder")
        right_shoulder = self._point(pose, "right_shoulder")
        shoulder_center = midpoint(left_shoulder, right_shoulder)
        shoulder_width = distance(left_shoulder, right_shoulder)

        has_left_hip = self._visible(pose, "left_hip", 0.22)
        has_right_hip = self._visible(pose, "right_hip", 0.22)
        has_hips = has_left_hip and has_right_hip
        left_hip = self._point(pose, "left_hip") if has_left_hip else None
        right_hip = self._point(pose, "right_hip") if has_right_hip else None
        hip_center = midpoint(left_hip, right_hip) if has_hips else None
        hip_width = distance(left_hip, right_hip) if has_hips else 0.0
        torso_height = distance(shoulder_center, hip_center) if has_hips else shoulder_width * 1.7

        xs = np.array([point[0] for point in visible_points.values()])
        ys = np.array([point[1] for point in visible_points.values()])
        body_height = float(ys.max() - ys.min())
        body_center_x = float((xs.min() + xs.max()) / 2.0)
        metrics = {
            "mode": mode,
            "shoulder_width": round(shoulder_width, 2),
            "hip_width": round(hip_width, 2) if has_hips else None,
            "torso_height": round(torso_height, 2),
            "body_center_x": round(body_center_x, 2),
            "body_height": round(body_height, 2),
        }

        if shoulder_width < width * 0.07:
            return self._fail(
                "Person is too far away. Move closer and keep your shoulders visible.",
                "TOO_FAR",
                ["Shoulder width is too small for reliable fitting"],
                metrics,
            )

        if shoulder_width > width * 0.68:
            return self._fail(
                "Person is too close or cropped. Step back and keep your upper body in frame.",
                "TOO_CLOSE",
                ["Shoulder width is too large for the image"],
                metrics,
            )

        if shoulder_center[1] < height * 0.08:
            return self._fail(
                "Leave a little space above your shoulders and retake the photo.",
                "TOO_CLOSE",
                ["Shoulders are too close to the top edge"],
                metrics,
            )

        center_offset = abs(shoulder_center[0] - (width / 2.0))
        if center_offset > width * 0.28:
            return self._fail(
                "Center yourself in the frame and retake the photo.",
                "UPPER_BODY_NOT_CENTERED",
                ["Upper body center is too far from the image center"],
                metrics,
            )

        face_visible = self._visible(pose, "nose", 0.25) or (
            self._visible(pose, "left_eye", 0.25) and self._visible(pose, "right_eye", 0.25)
        )
        if not face_visible:
            return self._fail(
                "Face the camera clearly and retake the photo.",
                "NOT_FRONT_FACING",
                ["No clear nose or eye landmarks detected"],
                metrics,
            )

        if has_hips:
            width_ratio = shoulder_width / max(hip_width, 1.0)
            metrics["shoulder_to_hip_width_ratio"] = round(width_ratio, 2)
            if width_ratio < 0.55 or width_ratio > 2.20:
                return self._fail(
                    "Stand front-facing with shoulders square to the camera.",
                    "NOT_FRONT_FACING",
                    ["Shoulder-to-hip width ratio suggests a turned pose"],
                    metrics,
                )

            vertical_tilt = abs(shoulder_center[0] - hip_center[0])
            metrics["vertical_tilt"] = round(float(vertical_tilt), 2)
            if vertical_tilt > shoulder_width * 0.85:
                return self._fail(
                    "Stand upright and face the camera directly.",
                    "NOT_FRONT_FACING",
                    ["Shoulder and hip centers are too horizontally offset"],
                    metrics,
                )

        if mode == "full_body":
            if not has_hips:
                return self._fail(
                    "Show your full body from shoulders to legs and retake the photo.",
                    "LOWER_BODY_NOT_VISIBLE",
                    ["Hip landmarks are required for full-body mode"],
                    metrics,
                )

            knees_visible = self._visible(pose, "left_knee", 0.22) and self._visible(pose, "right_knee", 0.22)
            ankles_visible = self._visible(pose, "left_ankle", 0.22) and self._visible(pose, "right_ankle", 0.22)
            if not (knees_visible or ankles_visible):
                return self._fail(
                    "Full-body mode needs your legs visible. Step back until knees or ankles are in frame.",
                    "LOWER_BODY_NOT_VISIBLE",
                    ["No reliable knee or ankle landmarks detected"],
                    metrics,
                )

            lower_names = ["left_hip", "right_hip"]
            lower_names.extend(["left_ankle", "right_ankle"] if ankles_visible else ["left_knee", "right_knee"])
            lower_points = [self._point(pose, name) for name in lower_names]
            lower_y_max = max(point[1] for point in lower_points)
            if lower_y_max < height * 0.62:
                return self._fail(
                    "Step back until more of your lower body is visible.",
                    "LOWER_BODY_NOT_VISIBLE",
                    ["Lower-body landmarks end too high in the image"],
                    metrics,
                )

        anchors = {
            "shoulder_width": shoulder_width,
            "shoulder_center": shoulder_center,
            "torso_height": torso_height,
            "hip_width": hip_width if has_hips else None,
            "hip_center": hip_center,
            "has_hips": has_hips,
            "body_center_x": body_center_x,
        }

        return PoseValidationResult(
            True,
            "Pose looks usable.",
            error_code=None,
            details=[],
            metrics=metrics,
            anchors=anchors,
        )
