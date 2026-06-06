from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ASSETS_DIR = PROJECT_ROOT / "assets"
ARMY_ASSET_DIR = ASSETS_DIR / "army"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)

CAMERA_INDEX = 0

WINDOW_NAME = "Army Uniform Overlay"

DEFAULT_UNIFORM = "army"

VALID_VIEWS = ["front", "back", "left", "right"]

VISIBILITY_THRESHOLD = 0.40
FACE_VISIBILITY_THRESHOLD = 0.45

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SHOW_LANDMARKS = True