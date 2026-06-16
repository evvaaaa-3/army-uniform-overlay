from dataclasses import dataclass
import math
import time

import cv2

from src.capture.stability_tracker import StabilityTracker
from src.config.settings import (
    AUTO_CAPTURE_COOLDOWN_SECONDS,
    AUTO_CAPTURE_COUNTDOWN_SECONDS,
    AUTO_CAPTURE_ENABLED,
    AUTO_CAPTURE_STABLE_SECONDS,
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAX_CAPTURE_MOTION_PX,
)
from src.tryon.pose_validation import FrontPoseValidator
from src.utils.file_utils import ORIGINALS_DIR, timestamped_path

try:
    import winsound
except ImportError:
    winsound = None


GUIDE_LINES = [
    "Q quit | A auto | SPACE/C manual capture | D debug | R reset",
]


@dataclass
class CaptureResult:
    image_path: str
    debug_enabled: bool


def _beep(frequency=900, duration_ms=140):
    if winsound is None:
        return
    try:
        winsound.Beep(frequency, duration_ms)
    except RuntimeError:
        pass


def get_guide_rect(frame_shape):
    h, w = frame_shape[:2]
    margin_x = int(w * 0.16)
    top_y = int(h * 0.08)
    bottom_y = int(h * 0.94)
    return margin_x, top_y, w - margin_x, bottom_y


def draw_capture_guide(frame, color=(0, 255, 255)):
    x1, y1, x2, y2 = get_guide_rect(frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def _draw_text(frame, text, origin, scale=0.75, color=(0, 255, 255), thickness=2):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_center_text(frame, text, y, scale=1.5, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
    x = max(20, int((w - text_size[0]) / 2))
    _draw_text(frame, text, (x, y), scale, color, 3)


def _pose_bounds(pose, min_visibility=0.35):
    points = [
        point for point in pose.values()
        if point.get("vis", 0.0) >= min_visibility
    ]
    if not points:
        return None
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _guide_frame_status(pose, frame_shape):
    if pose is None:
        return False, "No person detected"

    bounds = _pose_bounds(pose)
    if bounds is None:
        return False, "No person detected"

    gx1, gy1, gx2, gy2 = get_guide_rect(frame_shape)
    px1, py1, px2, py2 = bounds
    body_h = py2 - py1
    frame_h = frame_shape[0]

    if px1 < gx1 or px2 > gx2:
        return False, "Move into the box"
    if py1 < gy1 or py2 > gy2:
        return False, "Show full body"

    pose_center_x = (px1 + px2) / 2.0
    guide_center_x = (gx1 + gx2) / 2.0
    if abs(pose_center_x - guide_center_x) > (gx2 - gx1) * 0.18:
        return False, "Move into the box"

    if body_h > frame_h * 0.84:
        return False, "Step back"
    if body_h < frame_h * 0.48:
        return False, "Move closer"

    return True, "Hold still"


def _save_clean_capture(frame):
    captured_path = timestamped_path(ORIGINALS_DIR, "person", ".png")
    if not cv2.imwrite(str(captured_path), frame):
        raise RuntimeError(f"Could not save captured image: {captured_path}")
    print("[SAVED ORIGINAL]", captured_path)
    return str(captured_path)


def capture_photo(detector=None, validator=None, window_name="Photo Uniform Capture"):
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    if detector is None:
        from src.pose.pose_detector import PoseDetector
        detector = PoseDetector()

    validator = validator or FrontPoseValidator()
    stability = StabilityTracker(
        required_stable_seconds=AUTO_CAPTURE_STABLE_SECONDS,
        max_motion_px=MAX_CAPTURE_MOTION_PX,
    )

    print("Camera opened.")
    print("Auto-capture: ON")
    print("A: toggle auto-capture")
    print("SPACE or C: manual capture")
    print("D: toggle debug image output")
    print("R: reset after capture")
    print("Q: quit")

    auto_capture_enabled = AUTO_CAPTURE_ENABLED
    debug_enabled = False
    countdown_started_at = None
    captured_at = None
    captured_path = None
    last_countdown_value = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Could not read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            clean_frame = frame.copy()
            preview = frame.copy()

            pose, _ = detector.detect(frame)
            validation = validator.validate(pose, frame.shape)
            guide_ok, guide_message = _guide_frame_status(pose, frame.shape)
            now = time.time()

            if validation.ok and guide_ok and captured_path is None:
                stability_state = stability.update(pose, now)
            else:
                stability.reset()
                stability_state = {"stable": False, "stable_for": 0.0, "motion": 0.0}

            ready = (
                auto_capture_enabled
                and captured_path is None
                and validation.ok
                and guide_ok
                and stability_state["stable"]
            )

            if not ready:
                countdown_started_at = None
                last_countdown_value = None

            if ready and countdown_started_at is None:
                countdown_started_at = now
                last_countdown_value = int(AUTO_CAPTURE_COUNTDOWN_SECONDS)
                _beep(800, 120)

            main_message = guide_message
            guide_color = (0, 255, 255)

            if pose is not None and not validation.ok:
                if "Front pose" in validation.message or "sideways" in validation.message:
                    main_message = "Face front"
                elif "Full body" in validation.message or "ankles" in validation.message:
                    main_message = "Show full body"
                elif "too close" in validation.message:
                    main_message = "Step back"
                elif "too small" in validation.message:
                    main_message = "Move closer"
                else:
                    main_message = validation.message
                guide_color = (0, 0, 255)

            if validation.ok and guide_ok and not stability_state["stable"] and captured_path is None:
                main_message = "Hold still"

            if countdown_started_at is not None and captured_path is None:
                elapsed = now - countdown_started_at
                remaining = AUTO_CAPTURE_COUNTDOWN_SECONDS - elapsed
                if remaining > 0:
                    countdown_value = max(1, math.ceil(remaining))
                    main_message = f"Capturing in {countdown_value}..."
                    if countdown_value != last_countdown_value:
                        _beep(850, 100)
                        last_countdown_value = countdown_value
                else:
                    captured_path = _save_clean_capture(clean_frame)
                    captured_at = now
                    main_message = "Captured successfully"
                    _beep(1200, 180)

            if captured_path is not None:
                main_message = "Captured successfully"
                guide_color = (0, 180, 0)

            draw_capture_guide(preview, guide_color)

            status_lines = [
                f"Auto-capture: {'ON' if auto_capture_enabled else 'OFF'}",
                f"Debug: {'ON' if debug_enabled else 'OFF'}",
                f"Stable: {stability_state['stable_for']:.1f}s / {AUTO_CAPTURE_STABLE_SECONDS:.1f}s",
                f"Motion: {stability_state['motion']:.1f}px",
            ]
            y = 30
            for line in status_lines:
                _draw_text(preview, line, (20, y), 0.62, (0, 255, 255), 2)
                y += 26

            _draw_center_text(preview, main_message, int(preview.shape[0] * 0.18), 1.2, guide_color)

            if countdown_started_at is not None and captured_path is None:
                _draw_center_text(preview, "READY", int(preview.shape[0] * 0.30), 1.5, (0, 255, 0))

            controls_y = preview.shape[0] - 58
            for line in GUIDE_LINES:
                _draw_text(preview, line, (20, controls_y), 0.62, (255, 255, 255), 2)
                controls_y += 28

            if captured_path:
                _draw_text(preview, "R reset and retake | Q quit", (20, preview.shape[0] - 24), 0.62, (0, 255, 0), 2)

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), 27]:
                break

            if key == ord("a"):
                auto_capture_enabled = not auto_capture_enabled
                countdown_started_at = None
                print(f"[AUTO CAPTURE] {'ON' if auto_capture_enabled else 'OFF'}")

            if key == ord("d"):
                debug_enabled = not debug_enabled
                print(f"[DEBUG] {'ON' if debug_enabled else 'OFF'}")

            if key == ord("r"):
                captured_path = None
                captured_at = None
                countdown_started_at = None
                stability.reset()
                print("[RESET] Ready to capture again.")

            if key in [ord("c"), 32]:
                captured_path = _save_clean_capture(clean_frame)
                captured_at = now
                _beep(1200, 180)

            if captured_path is not None and captured_at is not None:
                if now - captured_at >= AUTO_CAPTURE_COOLDOWN_SECONDS:
                    return CaptureResult(captured_path, debug_enabled)

                if key not in [ord("r")]:
                    continue

            if captured_path is not None and key not in [ord("r")]:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_path is None:
        return None
    return CaptureResult(captured_path, debug_enabled)
