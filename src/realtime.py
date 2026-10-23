import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import create_hands, extract_features, _mp_hands

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    model_path = Path("models/isl_svm.joblib")
    labels_path = Path("models/labels.json")
    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError("Train the model first with: python src/train.py")

    model = joblib.load(model_path)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    hands = create_hands(static_image_mode=False)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        hands.close()
        raise RuntimeError("Could not open webcam. Try --camera 1 or check camera permissions.")

    print("Webcam started. Hold a supported static gesture. Press q to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            features = extract_features(frame, hands)

            prediction = "No hand detected"
            confidence = None
            if features is not None:
                prediction = str(model.predict([features])[0])
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([features])[0]
                    confidence = float(np.max(probs))

            cv2.rectangle(frame, (10, 10), (430, 90), (20, 20, 20), -1)
            cv2.putText(frame, f"Prediction: {prediction}",
                        (20, 45), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
            if confidence is not None:
                cv2.putText(frame, f"Confidence: {confidence:.1%}",
                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(frame, "Press q to quit",
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("ISL Static Gesture Recognition - SVM", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
