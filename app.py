from pathlib import Path
import json
import sys
import threading

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODEL_PATH = ROOT / "models" / "isl_svm.joblib"
LABELS_PATH = ROOT / "models" / "labels.json"

sys.path.insert(0, str(SRC))
from src.features import create_hands, extract_features  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

if not MODEL_PATH.exists() or not LABELS_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run 'python src/train.py' first."
    )

model = joblib.load(MODEL_PATH)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = json.load(f)

# MediaPipe Hands instances are reused for speed. A lock keeps inference safe
# if multiple browser requests arrive close together.
static_hands = create_hands(static_image_mode=True)
live_hands = create_hands(static_image_mode=True)
inference_lock = threading.Lock()


def predict_image(image_bgr, live=False):
    hands = live_hands if live else static_hands
    with inference_lock:
        features = extract_features(image_bgr, hands)

    if features is None:
        return {"success": False, "message": "No hand detected. Please show one clear hand."}

    prediction = str(model.predict([features])[0])
    confidence = None
    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([features])[0]
        confidence = float(np.max(probs))
        top_indices = np.argsort(probs)[::-1][:3]
        probabilities = [
            {"label": str(labels[i]), "confidence": float(probs[i])}
            for i in top_indices
        ]

    return {
        "success": True,
        "prediction": prediction,
        "confidence": confidence,
        "top_predictions": probabilities or [],
    }


@app.get("/")
def index():
    return render_template("index.html", class_count=len(labels))


@app.post("/api/predict-upload")
def predict_upload():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify({"success": False, "message": "Please choose an image."}), 400

    data = uploaded.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"success": False, "message": "The uploaded file is not a valid image."}), 400

    return jsonify(predict_image(image, live=False))


@app.post("/api/predict-frame")
def predict_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image", "")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    if not image_data:
        return jsonify({"success": False, "message": "No camera frame received."}), 400

    import base64
    try:
        raw = base64.b64decode(image_data)
    except Exception:
        return jsonify({"success": False, "message": "Invalid camera frame."}), 400

    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"success": False, "message": "Could not decode camera frame."}), 400

    return jsonify(predict_image(image, live=True))


@app.get("/api/status")
def status():
    return jsonify({
        "model_loaded": True,
        "classes": len(labels),
        "mode": "Static ISL + SVM",
    })


if __name__ == "__main__":
    print("\nISL Static Gesture Recognition Web App")
    print("Open http://127.0.0.1:5000 in your browser.\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
