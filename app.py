from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import os
import time
import threading
from datetime import datetime

from src.pose.pose_detector import PoseDetector
from src.pose.body_mapper import BodyMapper
from src.garments.asset_loader import GarmentLibrary
from src.garments.view_selector import select_view
from src.overlay.renderer import UniformRenderer
from src.config.settings import ARMY_ASSET_DIR, VISIBILITY_THRESHOLD
from src.tryon.photo_uniform_pipeline import PhotoUniformPipeline
from src.utils.file_utils import ORIGINALS_DIR, ensure_photo_dirs, timestamped_path

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
    save_debug=True,
)
print("All models loaded. Server ready.")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ensure_photo_dirs()

def auto_delete(filepath, delay=60):
    """Delete file after delay seconds for privacy."""
    def delete():
        time.sleep(delay)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Auto deleted: {filepath}")
    threading.Thread(target=delete, daemon=True).start()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "photo_pipeline": "ready"})

@app.route('/process-photo', methods=['POST'])
def process_photo():
    try:
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "Upload an image file using multipart field name 'image'.",
            }), 400

        uploaded = request.files['image']
        data = uploaded.read()
        if not data:
            return jsonify({"success": False, "message": "Uploaded image is empty."}), 400

        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"success": False, "message": "Could not decode uploaded image."}), 400

        original_path = timestamped_path(ORIGINALS_DIR, "upload", ".jpg")
        if not cv2.imwrite(str(original_path), frame):
            return jsonify({"success": False, "message": "Could not save uploaded original image."}), 500

        result = photo_pipeline.process(original_path)
        status = 200 if result.success else 400

        return jsonify({
            "success": result.success,
            "message": result.message,
            "original_image_path": str(original_path),
            "output_image_path": result.output_path,
            "debug_image_path": result.debug_path,
        }), status

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/overlay', methods=['POST'])
def overlay():
    try:
        data = request.get_json()
        image_data = data['image']

        # Decode base64 image from phone
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Run overlay pipeline
        pose, results = detector.detect(frame)

        if pose is None:
            return jsonify({"error": "No body detected. Please stand back so full body is visible."}), 400

        active_view = select_view(pose, "front")
        output = renderer.render(frame, pose, active_view)

        # Save result temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, output)

        # Auto delete after 60 seconds
        auto_delete(filepath, delay=60)

        # Return base64 result to phone
        with open(filepath, 'rb') as f:
            result_b64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({
            "success": True,
            "image": result_b64,
            "filename": filename
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
