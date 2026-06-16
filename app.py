from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

from src.pose.pose_detector import PoseDetector
from src.pose.body_mapper import BodyMapper
from src.garments.asset_loader import GarmentLibrary
from src.garments.view_selector import select_view
from src.overlay.renderer import UniformRenderer
from src.config.settings import ARMY_ASSET_DIR, PROJECT_ROOT, VISIBILITY_THRESHOLD
from src.tryon.photo_uniform_pipeline import PhotoUniformPipeline
from src.utils.file_utils import DEBUG_OUTPUT_DIR, FINAL_OUTPUT_DIR, ORIGINALS_DIR, ensure_photo_dirs, timestamped_path

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "5050"))
OUTPUT_EXPIRY_SECONDS = int(os.getenv("OUTPUT_EXPIRY_SECONDS", "3600"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

app = Flask(__name__)
CORS(app)

# Load all models once when server starts
print("Loading pose detector...")
detector = PoseDetector()
body_mapper = BodyMapper(VISIBILITY_THRESHOLD)
garment_library = GarmentLibrary(ARMY_ASSET_DIR)
renderer = UniformRenderer(garment_library, body_mapper)
photo_pipeline = PhotoUniformPipeline(
    detector=detector,
    garment_library=garment_library,
    body_mapper=body_mapper,
    renderer=renderer,
    save_debug=False,
)
print("All models loaded. Server ready.")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
ensure_photo_dirs()
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _allowed_cleanup_roots():
    return [
        ORIGINALS_DIR.resolve(),
        FINAL_OUTPUT_DIR.resolve(),
        DEBUG_OUTPUT_DIR.resolve(),
        OUTPUT_DIR.resolve(),
    ]


def _is_safe_cleanup_path(path):
    try:
        resolved = Path(path).resolve()
    except (TypeError, OSError):
        return False
    return any(resolved == root or root in resolved.parents for root in _allowed_cleanup_roots())


def schedule_cleanup(paths, delay_seconds=OUTPUT_EXPIRY_SECONDS):
    if paths is None:
        return
    if isinstance(paths, (str, Path)):
        paths = [paths]

    def delete():
        time.sleep(delay_seconds)
        for path in paths:
            if not _is_safe_cleanup_path(path):
                print(f"Skipped unsafe cleanup path: {path}")
                continue
            cleanup_path = Path(path)
            try:
                if cleanup_path.exists() and cleanup_path.is_file():
                    cleanup_path.unlink()
                    print(f"Auto deleted: {cleanup_path}")
            except OSError as exc:
                print(f"Could not auto delete {cleanup_path}: {exc}")

    threading.Thread(target=delete, daemon=True).start()


def auto_delete(filepath, delay=OUTPUT_EXPIRY_SECONDS):
    """Backward-compatible wrapper for legacy code paths."""
    schedule_cleanup(filepath, delay)


def _public_url(relative_url):
    if not relative_url or not PUBLIC_BASE_URL:
        return relative_url
    return f"{PUBLIC_BASE_URL.rstrip('/')}/{relative_url.lstrip('/')}"


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "success": True,
        "service": "army-uniform-overlay",
        "message": "Backend is running. Use /health or POST /process-photo.",
        "health_url": _public_url("/health"),
        "process_url": _public_url("/process-photo"),
        "overlay_url": _public_url("/overlay"),
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "success": True,
        "status": "ok",
        "service": "army-uniform-overlay",
        "version": "demo-photo-mvp",
        "photo_pipeline": "ready",
        "supports": ["upper_body", "full_body"],
        "legacy_overlay": True,
    })


def _json_error(message, status=400, error_code="PROCESSING_ERROR", details=None):
    return jsonify({
        "success": False,
        "message": message,
        "error_code": error_code,
        "details": details or [],
    }), status


def _is_allowed_upload(file_storage):
    filename = secure_filename(file_storage.filename or "")
    suffix = Path(filename).suffix.lower()
    mimetype = (file_storage.mimetype or "").lower()
    return suffix in ALLOWED_IMAGE_EXTENSIONS or mimetype.startswith("image/")


def _relative_output_url(path, output_type="final"):
    if path is None:
        return None
    filename = Path(path).name
    return _public_url(f"/outputs/{output_type}/{filename}")

@app.route('/process-photo', methods=['POST'])
def process_photo():
    try:
        if 'image' not in request.files:
            return _json_error(
                "Upload an image file using multipart field name 'image'.",
                400,
                "MISSING_IMAGE",
                ["Multipart form field 'image' was not found"],
            )

        uploaded = request.files['image']
        if uploaded.filename == "":
            return _json_error("Uploaded image filename is empty.", 400, "MISSING_IMAGE")

        if not _is_allowed_upload(uploaded):
            return _json_error(
                "Upload a valid image file.",
                400,
                "INVALID_IMAGE",
                ["Only common image uploads are accepted"],
            )

        mode = request.form.get("mode", "upper_body").strip().lower()
        if mode not in {"upper_body", "full_body"}:
            return _json_error(
                "Mode must be 'upper_body' or 'full_body'.",
                400,
                "INVALID_MODE",
                [f"Received mode: {mode}"],
            )
        debug = request.form.get("debug", "false").strip().lower() == "true"

        data = uploaded.read()
        if not data:
            return _json_error("Uploaded image is empty.", 400, "INVALID_IMAGE")

        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return _json_error(
                "Could not decode uploaded image.",
                400,
                "INVALID_IMAGE",
                ["OpenCV could not decode the uploaded bytes"],
            )

        original_path = timestamped_path(ORIGINALS_DIR, "upload", ".png")
        if not cv2.imwrite(str(original_path), frame):
            return _json_error("Could not save uploaded original image.", 500, "PROCESSING_ERROR")

        output_path = timestamped_path(FINAL_OUTPUT_DIR, "uniform_tryon", ".png")
        debug_path = timestamped_path(DEBUG_OUTPUT_DIR, "debug", ".png") if debug else None
        result = photo_pipeline.process(
            original_path,
            output_path=output_path,
            debug_path=debug_path,
            mode=mode,
            debug=debug,
            demo_mode=DEMO_MODE,
        )
        status = 200 if result.success else 400

        if not result.success:
            cleanup_paths = [original_path]
            if result.debug_path:
                cleanup_paths.append(result.debug_path)
            schedule_cleanup(cleanup_paths)
            return _json_error(
                result.message,
                status,
                result.error_code or "PROCESSING_ERROR",
                result.details,
            )

        final_filename = Path(result.output_path).name
        cleanup_paths = [result.output_path, original_path]
        if result.debug_path:
            cleanup_paths.append(result.debug_path)
        schedule_cleanup(cleanup_paths)

        payload = {
            "success": True,
            "message": "Uniform try-on image generated successfully.",
            "mode": mode,
            "image_url": _public_url(f"/outputs/final/{final_filename}"),
            "download_url": _public_url(f"/download/{final_filename}"),
            "expires_in_seconds": OUTPUT_EXPIRY_SECONDS,
            "debug_url": _relative_output_url(result.debug_path, "debug"),
        }
        if result.demo_fallback:
            payload["demo_fallback"] = True

        return jsonify(payload), status

    except Exception as e:
        return _json_error("Photo processing failed.", 500, "PROCESSING_ERROR", [str(e)])


@app.route('/outputs/final/<path:filename>', methods=['GET'])
def serve_final_output(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename:
        return _json_error("Invalid output filename.", 400, "INVALID_FILENAME")
    if not (FINAL_OUTPUT_DIR / safe_name).is_file():
        return _json_error("Output image not found.", 404, "NOT_FOUND")
    return send_from_directory(FINAL_OUTPUT_DIR, safe_name)


@app.route('/outputs/debug/<path:filename>', methods=['GET'])
def serve_debug_output(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename:
        return _json_error("Invalid debug filename.", 400, "INVALID_FILENAME")
    if not (DEBUG_OUTPUT_DIR / safe_name).is_file():
        return _json_error("Debug image not found.", 404, "NOT_FOUND")
    return send_from_directory(DEBUG_OUTPUT_DIR, safe_name)


@app.route('/download/<path:filename>', methods=['GET'])
def download_output(filename):
    safe_name = secure_filename(filename)
    if safe_name != filename:
        return _json_error("Invalid output filename.", 400, "INVALID_FILENAME")
    if not (FINAL_OUTPUT_DIR / safe_name).is_file():
        return _json_error("Output image not found.", 404, "NOT_FOUND")
    return send_from_directory(FINAL_OUTPUT_DIR, safe_name, as_attachment=True)


@app.route('/latest', methods=['GET'])
def latest_output():
    candidates = [
        path for path in FINAL_OUTPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]
    if not candidates:
        return _json_error("No final output image found.", 404, "NOT_FOUND")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return jsonify({
        "success": True,
        "image_url": _public_url(f"/outputs/final/{latest.name}"),
        "download_url": _public_url(f"/download/{latest.name}"),
    })

@app.route('/overlay', methods=['POST'])
def overlay():
    try:
        print("[/overlay] Request received.")
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            print("[/overlay] Bad request: body is not valid JSON.")
            return jsonify({
                "success": False,
                "error": "Request body must be JSON with an 'image' field.",
            }), 400

        image_data = data.get("image")
        if not image_data or not isinstance(image_data, str):
            print("[/overlay] Bad request: missing or invalid 'image' field.")
            return jsonify({
                "success": False,
                "error": "Missing base64 image field.",
            }), 400

        if image_data.startswith("data:"):
            if "," not in image_data:
                print("[/overlay] Bad request: malformed data URL image prefix.")
                return jsonify({
                    "success": False,
                    "error": "Malformed data URL image.",
                }), 400
            image_data = image_data.split(",", 1)[1]

        image_data = "".join(image_data.split())
        missing_padding = len(image_data) % 4
        if missing_padding:
            image_data += "=" * (4 - missing_padding)

        try:
            img_bytes = base64.b64decode(image_data, validate=True)
        except Exception as exc:
            print(f"[/overlay] Bad request: base64 decode failed: {exc}")
            return jsonify({
                "success": False,
                "error": "Invalid base64 image data.",
            }), 400

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[/overlay] Bad request: OpenCV could not decode image bytes.")
            return jsonify({
                "success": False,
                "error": "Could not decode image.",
            }), 400

        original_path = timestamped_path(ORIGINALS_DIR, "overlay_upload", ".png")
        if not cv2.imwrite(str(original_path), frame):
            print(f"[/overlay] Server error: could not save original image to {original_path}.")
            return jsonify({
                "success": False,
                "error": "Could not save uploaded image.",
            }), 500

        output_path = timestamped_path(FINAL_OUTPUT_DIR, "overlay_result", ".png")
        result = photo_pipeline.process(
            original_path,
            output_path=output_path,
            mode="upper_body",
            debug=False,
            demo_mode=DEMO_MODE,
        )

        if not result.success:
            print(f"[/overlay] Processing failed: {result.error_code} {result.message} {result.details}")
            schedule_cleanup([original_path, result.debug_path])
            return jsonify({
                "success": False,
                "error": result.message,
            }), 400

        with open(result.output_path, "rb") as f:
            result_b64 = base64.b64encode(f.read()).decode('utf-8')

        cleanup_paths = [original_path, result.output_path]
        if result.debug_path:
            cleanup_paths.append(result.debug_path)
        schedule_cleanup(cleanup_paths)

        print(f"[/overlay] Success: generated {Path(result.output_path).name}.")
        return jsonify({
            "success": True,
            "image": result_b64,
        })

    except Exception as e:
        print(f"[/overlay] Server error: {e}")
        return jsonify({
            "success": False,
            "error": "Overlay processing failed.",
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host=APP_HOST, port=APP_PORT)
