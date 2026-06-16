from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from src.config.settings import ARMY_ASSET_DIR, VISIBILITY_THRESHOLD
from src.garments.asset_loader import GarmentLibrary
from src.overlay.renderer import UniformRenderer
from src.pose.body_mapper import BodyMapper
from src.pose.pose_detector import PoseDetector
from src.tryon.pose_validation import FrontPoseValidator
from src.utils.file_utils import DEBUG_OUTPUT_DIR, FINAL_OUTPUT_DIR, ensure_photo_dirs, timestamped_path


@dataclass
class PhotoTryOnResult:
    success: bool
    message: str
    output_path: Optional[str] = None
    debug_path: Optional[str] = None


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

    def process(self, image_path, output_path=None, debug_path=None):
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if frame is None:
            return PhotoTryOnResult(False, f"Could not read image: {image_path}")

        try:
            pose, _ = self.detector.detect(frame)
            validation = self.validator.validate(pose, frame.shape)

            if debug_path is None and self.save_debug:
                debug_path = timestamped_path(DEBUG_OUTPUT_DIR, "debug", ".jpg")

            saved_debug_path = None
            if self.save_debug and debug_path is not None:
                saved_debug_path = self._save_debug(frame, pose, validation, debug_path)

            if not validation.ok:
                return PhotoTryOnResult(
                    False,
                    validation.message,
                    debug_path=str(saved_debug_path) if saved_debug_path else None,
                )

            output = self.renderer.render(frame, pose, "front")

            if output_path is None:
                output_path = timestamped_path(FINAL_OUTPUT_DIR, "uniform_tryon", ".jpg")
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

            if not cv2.imwrite(str(output_path), output):
                return PhotoTryOnResult(False, f"Could not save final image: {output_path}")

            return PhotoTryOnResult(
                True,
                "Uniform try-on image generated successfully.",
                output_path=str(output_path),
                debug_path=str(saved_debug_path) if saved_debug_path else None,
            )

        except Exception as exc:
            return PhotoTryOnResult(False, f"Photo processing failed: {exc}")

    def _save_debug(self, frame, pose, validation, debug_path):
        debug = frame.copy()

        if pose:
            for name, point in pose.items():
                if point["vis"] >= VISIBILITY_THRESHOLD:
                    cv2.circle(debug, (point["x"], point["y"]), 4, (0, 255, 255), -1)
                    cv2.putText(
                        debug,
                        name.replace("_", " "),
                        (point["x"] + 5, point["y"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

        color = (0, 180, 0) if validation.ok else (0, 0, 255)
        cv2.putText(
            debug,
            validation.message,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(debug_path), debug):
            return debug_path
        return None
