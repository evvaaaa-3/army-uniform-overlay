from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.utils.geometry import distance, midpoint, to_np


@dataclass
class PoseValidationResult:
    ok: bool
    message: str
    anchors: Optional[dict] = None


class FrontPoseValidator:
    """
    Conservative validation for the still-photo MVP.

    YOLO pose cannot truly prove front vs back, so "front" means face keypoints
    are visible and the body geometry looks like a mostly upright, full-body pose.
    """

    def __init__(self, visibility_threshold=0.40):
        self.visibility_threshold = visibility_threshold

    def _visible(self, pose, name, threshold=None):
        threshold = self.visibility_threshold if threshold is None else threshold
        return pose is not None and name in pose and pose[name]["vis"] >= threshold

    def _require(self, pose, names, message):
        missing = [name for name in names if not self._visible(pose, name)]
        if missing:
            return PoseValidationResult(False, message)
        return None

    def validate(self, pose, image_shape):
        height, width = image_shape[:2]

        if pose is None:
            return PoseValidationResult(
                False,
                "No body detected. Please stand back so your full body is visible.",
            )

        core = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        failed = self._require(
            pose,
            core,
            "Shoulders or hips were not detected clearly. Please face the camera and retake the photo.",
        )
        if failed:
            return failed

        legs = ["left_knee", "right_knee", "left_ankle", "right_ankle"]
        failed = self._require(
            pose,
            legs,
            "Full body not detected. Please step back until both legs and ankles are visible.",
        )
        if failed:
            return failed

        face_points = ["nose", "left_eye", "right_eye"]
        if not all(self._visible(pose, name, 0.35) for name in face_points):
            return PoseValidationResult(
                False,
                "Front pose not detected. Please face the camera and retake the photo.",
            )

        pts = {name: to_np(pose[name]) for name in core + legs + face_points}

        shoulder_mid = midpoint(pts["left_shoulder"], pts["right_shoulder"])
        hip_mid = midpoint(pts["left_hip"], pts["right_hip"])
        ankle_mid = midpoint(pts["left_ankle"], pts["right_ankle"])

        shoulder_width = distance(pts["left_shoulder"], pts["right_shoulder"])
        hip_width = distance(pts["left_hip"], pts["right_hip"])
        torso_height = distance(shoulder_mid, hip_mid)
        body_height = float(max(pts[name][1] for name in pts) - min(pts[name][1] for name in pts))

        if shoulder_width < width * 0.08 or body_height < height * 0.45:
            return PoseValidationResult(
                False,
                "Person is too small in frame. Please move a little closer while keeping the full body visible.",
            )

        if shoulder_width > width * 0.62 or body_height > height * 0.94:
            return PoseValidationResult(
                False,
                "Person is too close or cropped. Please step back and keep the full body inside the frame.",
            )

        all_xy = np.array(list(pts.values()))
        margin_x = width * 0.025
        margin_y = height * 0.025
        if (
            all_xy[:, 0].min() < margin_x
            or all_xy[:, 0].max() > width - margin_x
            or all_xy[:, 1].min() < margin_y
            or all_xy[:, 1].max() > height - margin_y
        ):
            return PoseValidationResult(
                False,
                "Body is too close to the edge of the frame. Please center yourself and retake the photo.",
            )

        if torso_height < height * 0.10:
            return PoseValidationResult(
                False,
                "Torso landmarks are unstable. Please stand upright and retake the photo.",
            )

        width_ratio = shoulder_width / max(hip_width, 1.0)
        if width_ratio < 0.65 or width_ratio > 1.90:
            return PoseValidationResult(
                False,
                "Pose angle looks too sideways. Please face the camera directly.",
            )

        vertical_tilt = abs(shoulder_mid[0] - hip_mid[0])
        if vertical_tilt > shoulder_width * 0.75:
            return PoseValidationResult(
                False,
                "Pose is too tilted or turned. Please stand straight facing the camera.",
            )

        if ankle_mid[1] <= hip_mid[1]:
            return PoseValidationResult(
                False,
                "Leg landmarks are unstable. Please keep your full body visible and retake the photo.",
            )

        anchors = {
            "shoulder_width": shoulder_width,
            "shoulder_center": shoulder_mid,
            "torso_height": torso_height,
            "hip_width": hip_width,
            "hip_center": hip_mid,
            "leg_length": distance(hip_mid, ankle_mid),
            "ankle_center": ankle_mid,
        }

        return PoseValidationResult(True, "Front pose detected.", anchors)
