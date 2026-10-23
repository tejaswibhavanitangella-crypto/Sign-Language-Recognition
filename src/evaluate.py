import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from features import create_hands, extract_features

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_features(data_dir):
    data_dir = Path(data_dir)
    X, y = [], []
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    hands = create_hands(static_image_mode=True)
    try:
        for label in classes:
            for path in (data_dir / label).rglob("*"):
                if path.suffix.lower() not in IMAGE_EXTS:
                    continue
                image = cv2.imread(str(path))
                if image is None:
                    continue
                feat = extract_features(image, hands)
                if feat is not None:
                    X.append(feat)
                    y.append(label)
    finally:
        hands.close()
    return np.asarray(X), np.asarray(y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    model_path = "models/isl_svm.joblib"
    if not Path(model_path).exists():
        raise FileNotFoundError("Train the model first with: python src/train.py")

    model = joblib.load(model_path)
    X, y = load_features(args.data)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )
    pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(classification_report(y_test, pred, zero_division=0))

if __name__ == "__main__":
    main()
