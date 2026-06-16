import argparse
from pathlib import Path

from src.capture.photo_capture import capture_photo
from src.tryon.photo_uniform_pipeline import PhotoUniformPipeline
from src.utils.file_utils import FINAL_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Capture a photo and apply the army uniform overlay.")
    parser.add_argument(
        "--mode",
        choices=["upper_body", "full_body"],
        default="upper_body",
        help="Try-on mode. upper_body is recommended for the app MVP.",
    )
    args = parser.parse_args()

    print("Starting still-photo army uniform try-on.")
    print(f"Mode: {args.mode}")
    print("Loading photo processing pipeline...")
    pipeline = PhotoUniformPipeline(save_debug=False)

    while True:
        capture = capture_photo(
            detector=pipeline.detector,
            validator=pipeline.validator,
        )
        if capture is None:
            print("No photo captured. Exiting.")
            return

        image_path = Path(capture.image_path)
        pipeline.save_debug = capture.debug_enabled
        timestamp = image_path.stem.replace("person_", "", 1)
        output_path = FINAL_OUTPUT_DIR / f"final_{timestamp}.png"

        print("Processing captured photo...")
        result = pipeline.process(image_path, output_path=output_path, mode=args.mode)

        if result.success:
            print("[SUCCESS]", result.message)
            print("[FINAL IMAGE]", result.output_path)
            if result.debug_path:
                print("[DEBUG IMAGE]", result.debug_path)
            return

        print("[RETAKE NEEDED]", result.message)
        if result.error_code:
            print("[ERROR CODE]", result.error_code)
        if result.details:
            print("[DETAILS]", "; ".join(result.details))
        if result.debug_path:
            print("[DEBUG IMAGE]", result.debug_path)
        print("Press SPACE/C in the preview to capture another photo, or Q to quit.")


if __name__ == "__main__":
    main()
