import time
from datetime import datetime

import cv2

from src.config.settings import (
    CAMERA_INDEX,
    WINDOW_NAME,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    SHOW_LANDMARKS,
    ARMY_ASSET_DIR,
    OUTPUTS_DIR,
    VISIBILITY_THRESHOLD,
)

from src.pose.pose_detector import PoseDetector
from src.pose.body_mapper import BodyMapper
from src.garments.asset_loader import GarmentLibrary
from src.garments.view_selector import select_view
from src.overlay.renderer import UniformRenderer


SCREENSHOT_INTERVAL = 10  # seconds


def save_frame(frame, prefix="modular_overlay"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUTS_DIR / f"{prefix}_{timestamp}.png"

    success = cv2.imwrite(str(path), frame)

    if success:
        print("[SAVED]", path)
    else:
        print("[ERROR] Could not save frame:", path)


def draw_hud(frame, active_view, forced_view, seconds_until_save):
    lines = [
        f"Active view: {active_view}",
        f"Forced view: {forced_view or 'auto'}",
        f"Auto screenshot in: {seconds_until_save}s",
        "Q quit | S save | A auto | 1 front | 2 back | 3 left | 4 right",
    ]

    y = 30

    for text in lines:
        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def main():
    print("Loading pose detector...")
    detector = PoseDetector()

    print("Loading body mapper...")
    body_mapper = BodyMapper(VISIBILITY_THRESHOLD)

    print("Loading garment assets from:", ARMY_ASSET_DIR)
    garment_library = GarmentLibrary(ARMY_ASSET_DIR)

    print("Creating renderer...")
    renderer = UniformRenderer(garment_library, body_mapper)

    print("Opening camera index:", CAMERA_INDEX)
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # Start with front view because auto detection was choosing back incorrectly.
    forced_view = "front"

    last_screenshot_time = time.time()

    print("\nControls:")
    print("Q quit")
    print("S save manually")
    print("A auto view")
    print("1 force front")
    print("2 force back")
    print("3 force left")
    print("4 force right")
    print(f"Auto screenshot every {SCREENSHOT_INTERVAL} seconds\n")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("[ERROR] Could not read frame from camera.")
            break

        # Mirror webcam feed for natural testing.
        frame = cv2.flip(frame, 1)

        output = frame.copy()

        pose, results = detector.detect(frame)

        active_view = "no_pose"

        if pose is not None:
            active_view = select_view(pose, forced_view)
            output = renderer.render(output, pose, active_view)

            if SHOW_LANDMARKS:
                detector.draw(output, results)

        current_time = time.time()
        elapsed = current_time - last_screenshot_time
        seconds_until_save = max(0, int(SCREENSHOT_INTERVAL - elapsed))

        draw_hud(output, active_view, forced_view, seconds_until_save)

        if elapsed >= SCREENSHOT_INTERVAL:
            save_frame(output, prefix=f"auto_{active_view}")
            last_screenshot_time = current_time

        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            save_frame(output, prefix=f"manual_{active_view}")

        elif key == ord("a"):
            forced_view = None
            print("[MODE] Auto view")

        elif key == ord("0"):
            forced_view = None
            print("[MODE] Auto view")

        elif key == ord("1"):
            forced_view = "front"
            print("[MODE] Forced front")

        elif key == ord("2"):
            forced_view = "back"
            print("[MODE] Forced back")

        elif key == ord("3"):
            forced_view = "left"
            print("[MODE] Forced left")

        elif key == ord("4"):
            forced_view = "right"
            print("[MODE] Forced right")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Exiting.")


if __name__ == "__main__":
    main()