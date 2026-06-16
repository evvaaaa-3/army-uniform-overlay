from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config.settings import ARMY_ASSET_DIR, VISIBILITY_THRESHOLD
from src.garments.asset_loader import GarmentLibrary
from src.overlay.renderer import UniformRenderer
from src.pose.body_mapper import BodyMapper
from src.pose.pose_detector import PoseDetector
from src.tryon.pose_validation import FrontPoseValidator
from src.utils.file_utils import DEBUG_OUTPUT_DIR, FINAL_OUTPUT_DIR, ensure_photo_dirs, timestamped_path
from src.utils.geometry import distance, midpoint, to_np


SHIRT_WIDTH_SCALE = 1.65
SHIRT_TOP_OFFSET_RATIO = 0.22
SHIRT_BOTTOM_OFFSET_RATIO = 0.18
SHIRT_MIN_WIDTH_RATIO = 0.22
SHIRT_MAX_WIDTH_RATIO = 0.65

TROUSER_WIDTH_SCALE = 1.55
TROUSER_TOP_OFFSET_RATIO = 0.08
TROUSER_BOTTOM_MARGIN_RATIO = 0.04

ALPHA_STRENGTH = 0.94
ALPHA_FEATHER_KERNEL = 9


@dataclass
class PhotoTryOnResult:
    success: bool
    message: str
    output_path: Optional[str] = None
    debug_path: Optional[str] = None
    error_code: Optional[str] = None
    details: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    demo_fallback: bool = False


class PhotoUniformPipeline:
    def __init__(
        self,
        detector=None,
        garment_library=None,
        body_mapper=None,
        renderer=None,
        validator=None,
        save_debug=False,
    ):
        self.detector = detector or PoseDetector()
        self.body_mapper = body_mapper or BodyMapper(VISIBILITY_THRESHOLD)
        self.garment_library = garment_library or GarmentLibrary(ARMY_ASSET_DIR)
        self.renderer = renderer or UniformRenderer(self.garment_library, self.body_mapper)
        self.validator = validator or FrontPoseValidator(VISIBILITY_THRESHOLD)
        self.save_debug = save_debug
        ensure_photo_dirs()

    def process(self, image_path, output_path=None, debug_path=None, mode="upper_body", debug=None, demo_mode=False):
        mode = mode if mode in {"upper_body", "full_body"} else "upper_body"
        debug_enabled = self.save_debug if debug is None else bool(debug)
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if frame is None:
            return PhotoTryOnResult(
                False,
                f"Could not read image: {image_path}",
                error_code="PROCESSING_ERROR",
                details=["OpenCV could not read the source image"],
            )

        try:
            pose, _ = self.detector.detect(frame)
            validation = self.validator.validate(pose, frame.shape, mode=mode)

            measurements = {}
            boxes = {}
            if validation.ok:
                measurements = self._measure_body(pose, frame.shape)
                boxes = self._fit_boxes(frame.shape, pose, measurements, mode)
                measurements.update(
                    {
                        "shirt_box": boxes.get("shirt"),
                        "trouser_box": boxes.get("trousers"),
                    }
                )

            if not validation.ok:
                if demo_mode:
                    measurements = self._fallback_measurements(frame.shape)
                    boxes = self._fit_boxes(frame.shape, pose, measurements, "upper_body")
                    measurements.update(
                        {
                            "shirt_box": boxes.get("shirt"),
                            "trouser_box": None,
                            "demo_fallback": True,
                        }
                    )
                    output = self._paste_garment(
                        frame.copy(),
                        self.garment_library.get("shirt", "front").image_bgra,
                        boxes["shirt"],
                    )

                    if output_path is None:
                        output_path = timestamped_path(FINAL_OUTPUT_DIR, "uniform_tryon", ".png")
                    else:
                        output_path = Path(output_path)
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                    saved_debug_path = None
                    if debug_path is None and debug_enabled:
                        debug_path = timestamped_path(DEBUG_OUTPUT_DIR, "debug", ".png")
                    if debug_enabled and debug_path is not None:
                        saved_debug_path = self._save_debug(frame, pose, validation, debug_path, mode, measurements, boxes)

                    if not cv2.imwrite(str(output_path), output):
                        return PhotoTryOnResult(False, f"Could not save final image: {output_path}", error_code="PROCESSING_ERROR")

                    return PhotoTryOnResult(
                        True,
                        "Uniform try-on image generated successfully.",
                        output_path=str(output_path),
                        debug_path=str(saved_debug_path) if saved_debug_path else None,
                        metrics=measurements,
                        demo_fallback=True,
                    )

                saved_debug_path = None
                if debug_path is None and debug_enabled:
                    debug_path = timestamped_path(DEBUG_OUTPUT_DIR, "debug", ".png")
                if debug_enabled and debug_path is not None:
                    saved_debug_path = self._save_debug(frame, pose, validation, debug_path, mode, measurements, boxes)
                return PhotoTryOnResult(
                    False,
                    validation.message,
                    debug_path=str(saved_debug_path) if saved_debug_path else None,
                    error_code=validation.error_code,
                    details=validation.details,
                    metrics=validation.metrics,
                )

            output = frame.copy()
            if mode == "full_body":
                trouser_box = boxes.get("trousers")
                if trouser_box is None:
                    return PhotoTryOnResult(
                        False,
                        "Full-body mode needs your legs visible. Step back until knees or ankles are in frame.",
                        debug_path=str(saved_debug_path) if saved_debug_path else None,
                        error_code="LOWER_BODY_NOT_VISIBLE",
                        details=["Could not calculate a trouser fit box from lower-body landmarks"],
                        metrics=measurements,
                    )
                output = self._paste_garment(output, self.garment_library.get("trousers", "front").image_bgra, trouser_box)

            output = self._paste_garment(output, self.garment_library.get("shirt", "front").image_bgra, boxes["shirt"])

            if output_path is None:
                output_path = timestamped_path(FINAL_OUTPUT_DIR, "uniform_tryon", ".png")
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(output_path), output):
                return PhotoTryOnResult(False, f"Could not save final image: {output_path}", error_code="PROCESSING_ERROR")

            saved_debug_path = None
            if debug_path is None and debug_enabled:
                debug_path = timestamped_path(DEBUG_OUTPUT_DIR, "debug", ".png")
            if debug_enabled and debug_path is not None:
                saved_debug_path = self._save_debug(frame, pose, validation, debug_path, mode, measurements, boxes)

            return PhotoTryOnResult(
                True,
                "Uniform try-on image generated successfully.",
                output_path=str(output_path),
                debug_path=str(saved_debug_path) if saved_debug_path else None,
                metrics=measurements,
            )

        except Exception as exc:
            return PhotoTryOnResult(
                False,
                "Photo processing failed.",
                error_code="PROCESSING_ERROR",
                details=[str(exc)],
            )

    def _visible(self, pose, name, threshold=0.22):
        return pose is not None and name in pose and pose[name]["vis"] >= threshold

    def _pt(self, pose, name):
        return to_np(pose[name])

    def _measure_body(self, pose, frame_shape):
        left_shoulder = self._pt(pose, "left_shoulder")
        right_shoulder = self._pt(pose, "right_shoulder")
        shoulder_center = midpoint(left_shoulder, right_shoulder)
        shoulder_width = distance(left_shoulder, right_shoulder)

        has_hips = self._visible(pose, "left_hip") and self._visible(pose, "right_hip")
        if has_hips:
            left_hip = self._pt(pose, "left_hip")
            right_hip = self._pt(pose, "right_hip")
            hip_center = midpoint(left_hip, right_hip)
            hip_width = distance(left_hip, right_hip)
            torso_height = max(distance(shoulder_center, hip_center), shoulder_width * 1.05)
        else:
            hip_center = np.array([shoulder_center[0], shoulder_center[1] + shoulder_width * 1.7], dtype=np.float32)
            hip_width = shoulder_width * 0.82
            torso_height = shoulder_width * 1.7

        visible_points = [
            to_np(point)
            for point in pose.values()
            if point.get("vis", 0.0) >= 0.22
        ]
        body_center_x = float(shoulder_center[0])
        if visible_points:
            xs = [point[0] for point in visible_points]
            body_center_x = float((min(xs) + max(xs)) / 2.0)

        return {
            "shoulder_center": shoulder_center,
            "shoulder_width": shoulder_width,
            "hip_center": hip_center,
            "hip_width": hip_width,
            "torso_height": torso_height,
            "body_center_x": body_center_x,
            "has_hips": has_hips,
        }

    def _fallback_measurements(self, frame_shape):
        height, width = frame_shape[:2]
        shoulder_width = max(width * 0.28, min(width, height) * 0.24)
        shoulder_center = np.array([width * 0.50, height * 0.34], dtype=np.float32)
        torso_height = shoulder_width * 1.7
        hip_center = np.array([shoulder_center[0], shoulder_center[1] + torso_height], dtype=np.float32)
        hip_width = shoulder_width * 0.82
        return {
            "shoulder_center": shoulder_center,
            "shoulder_width": shoulder_width,
            "hip_center": hip_center,
            "hip_width": hip_width,
            "torso_height": torso_height,
            "body_center_x": float(shoulder_center[0]),
            "has_hips": False,
        }

    def _fit_boxes(self, frame_shape, pose, measurements, mode):
        height, width = frame_shape[:2]
        shoulder_center = measurements["shoulder_center"]
        hip_center = measurements["hip_center"]
        shoulder_width = measurements["shoulder_width"]
        hip_width = measurements["hip_width"]
        torso_height = measurements["torso_height"]

        shirt_width = np.clip(
            shoulder_width * SHIRT_WIDTH_SCALE,
            width * SHIRT_MIN_WIDTH_RATIO,
            width * SHIRT_MAX_WIDTH_RATIO,
        )
        shirt_top_y = shoulder_center[1] - SHIRT_TOP_OFFSET_RATIO * torso_height
        shirt_bottom_y = shoulder_center[1] + torso_height + SHIRT_BOTTOM_OFFSET_RATIO * torso_height
        shirt_height = max(shirt_bottom_y - shirt_top_y, torso_height * 1.10)
        shirt_center_x = shoulder_center[0]
        shirt_box = self._box_from_center(
            shirt_center_x,
            (shirt_top_y + shirt_bottom_y) / 2.0,
            shirt_width,
            shirt_height,
            width,
            height,
        )

        boxes = {"shirt": shirt_box}

        if mode == "full_body":
            trouser_box = self._trouser_box(frame_shape, pose, hip_center, hip_width, torso_height)
            if trouser_box is not None:
                boxes["trousers"] = trouser_box

        return boxes

    def _trouser_box(self, frame_shape, pose, hip_center, hip_width, torso_height):
        height, width = frame_shape[:2]
        ankle_visible = self._visible(pose, "left_ankle") and self._visible(pose, "right_ankle")
        knee_visible = self._visible(pose, "left_knee") and self._visible(pose, "right_knee")

        if ankle_visible:
            left_lower = self._pt(pose, "left_ankle")
            right_lower = self._pt(pose, "right_ankle")
            lower_center = midpoint(left_lower, right_lower)
            lower_y = lower_center[1]
        elif knee_visible:
            left_lower = self._pt(pose, "left_knee")
            right_lower = self._pt(pose, "right_knee")
            lower_center = midpoint(left_lower, right_lower)
            lower_y = lower_center[1] + distance(hip_center, lower_center) * 0.95
        else:
            return None

        leg_height = max(lower_y - hip_center[1], torso_height * 1.2)
        trouser_width = max(hip_width * TROUSER_WIDTH_SCALE, width * 0.18)
        trouser_top_y = hip_center[1] - TROUSER_TOP_OFFSET_RATIO * torso_height
        trouser_bottom_y = lower_y + TROUSER_BOTTOM_MARGIN_RATIO * leg_height
        trouser_height = max(trouser_bottom_y - trouser_top_y, torso_height * 1.5)

        return self._box_from_center(
            hip_center[0],
            (trouser_top_y + trouser_bottom_y) / 2.0,
            min(trouser_width, width * 0.58),
            trouser_height,
            width,
            height,
        )

    def _box_from_center(self, center_x, center_y, box_width, box_height, image_width, image_height):
        box_width = int(round(max(8, box_width)))
        box_height = int(round(max(8, box_height)))
        x1 = int(round(center_x - box_width / 2.0))
        y1 = int(round(center_y - box_height / 2.0))
        x2 = x1 + box_width
        y2 = y1 + box_height

        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > image_width:
            x1 -= x2 - image_width
            x2 = image_width
        if y2 > image_height:
            y1 -= y2 - image_height
            y2 = image_height

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_width, max(x1 + 1, x2))
        y2 = min(image_height, max(y1 + 1, y2))
        return (x1, y1, x2, y2)

    def _paste_garment(self, frame_bgr, garment_bgra, box):
        x1, y1, x2, y2 = box
        target_w = max(1, x2 - x1)
        target_h = max(1, y2 - y1)
        resized = cv2.resize(garment_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if resized.shape[2] == 4:
            color = resized[:, :, :3].astype(np.float32)
            alpha = resized[:, :, 3].astype(np.float32) / 255.0
        else:
            color = resized.astype(np.float32)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            alpha = (gray < 245).astype(np.float32)

        alpha = self._clean_alpha(alpha) * ALPHA_STRENGTH
        alpha_3 = alpha[:, :, None]
        roi = frame_bgr[y1:y2, x1:x2].astype(np.float32)
        blended = color * alpha_3 + roi * (1.0 - alpha_3)
        frame_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame_bgr

    def _clean_alpha(self, alpha):
        alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        _, alpha_u8 = cv2.threshold(alpha_u8, 8, 255, cv2.THRESH_TOZERO)
        kernel_size = ALPHA_FEATHER_KERNEL if ALPHA_FEATHER_KERNEL % 2 == 1 else ALPHA_FEATHER_KERNEL + 1
        alpha_u8 = cv2.GaussianBlur(alpha_u8, (kernel_size, kernel_size), 0)
        return alpha_u8.astype(np.float32) / 255.0

    def _save_debug(self, frame, pose, validation, debug_path, mode, measurements, boxes):
        debug = frame.copy()

        if pose:
            for name, point in pose.items():
                if point["vis"] >= VISIBILITY_THRESHOLD:
                    cv2.circle(debug, (point["x"], point["y"]), 4, (0, 255, 255), -1)

        if pose and self._visible(pose, "left_shoulder") and self._visible(pose, "right_shoulder"):
            ls = tuple(self._pt(pose, "left_shoulder").astype(int))
            rs = tuple(self._pt(pose, "right_shoulder").astype(int))
            cv2.line(debug, ls, rs, (255, 255, 0), 2)

        if pose and self._visible(pose, "left_hip") and self._visible(pose, "right_hip"):
            lh = tuple(self._pt(pose, "left_hip").astype(int))
            rh = tuple(self._pt(pose, "right_hip").astype(int))
            cv2.line(debug, lh, rh, (0, 255, 0), 2)

        if boxes.get("shirt"):
            self._draw_box(debug, boxes["shirt"], (0, 180, 255), "shirt")
        if boxes.get("trousers"):
            self._draw_box(debug, boxes["trousers"], (255, 120, 0), "trousers")

        color = (0, 180, 0) if validation.ok else (0, 0, 255)
        lines = [
            f"mode: {mode}",
            validation.message,
            f"shoulder_width: {measurements.get('shoulder_width')}",
            f"hip_width: {measurements.get('hip_width')}",
            f"torso_height: {measurements.get('torso_height')}",
            f"shirt_box: {measurements.get('shirt_box')}",
            f"trouser_box: {measurements.get('trouser_box')}",
        ]
        y = 28
        for line in lines:
            cv2.putText(debug, str(line), (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            y += 24

        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(debug_path), debug):
            return debug_path
        return None

    def _draw_box(self, image, box, color, label):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
