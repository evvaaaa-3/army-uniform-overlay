from datetime import datetime
from pathlib import Path

from src.config.settings import PROJECT_ROOT


CAPTURES_DIR = PROJECT_ROOT / "captures"
ORIGINALS_DIR = CAPTURES_DIR / "originals"
FINAL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "final"
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "debug"


def ensure_photo_dirs():
    for directory in [ORIGINALS_DIR, FINAL_OUTPUT_DIR, DEBUG_OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def timestamped_path(directory: Path, prefix: str, suffix: str = ".jpg") -> Path:
    ensure_photo_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"{prefix}_{timestamp}{suffix}"
