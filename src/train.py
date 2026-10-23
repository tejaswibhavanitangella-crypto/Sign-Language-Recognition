import argparse
import json
import os
from pathlib import Path

import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from features import create_hands, extract_features

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_dataset(data_dir, images_per_class=200, random_state=42, exclude_classes=None):
    data_dir = Path(data_dir)
    X, y = [], []
    skipped = 0
    exclude_classes = set(exclude_classes or [])
    classes = sorted([p.name for p in data_dir.iterdir() if p.is_dir() and p.name not in exclude_classes])
    if not classes:
        raise FileNotFoundError(f"No class folders found in {data_dir}")

    print(f"Classes found ({len(classes)}): {classes}")
    hands = create_hands(static_image_mode=True)

    try:
        rng = np.random.default_rng(random_state)
        for label in classes:
            class_dir = data_dir / label
            image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            total = len(image_paths)
            if images_per_class > 0 and total > images_per_class:
                image_paths = image_paths[:images_per_class]
            print(f"Processing {label}: {len(image_paths)} of {total} images")
            for path in image_paths:
                image = cv2.imread(str(path))
                if image is None:
                    skipped += 1
                    continue
                feat = extract_features(image, hands)
                if feat is None:
                    skipped += 1
                    continue
                X.append(feat)
                y.append(label)

                # Instant exact data augmentation: negate relative X coordinates for left-hand support
                feat_flip = feat.copy()
                feat_flip[0::3] = -feat_flip[0::3]
                X.append(feat_flip)
                y.append(label)
    finally:
        hands.close()

    if not X:
        raise RuntimeError("No usable hand landmarks were extracted.")

    return np.asarray(X), np.asarray(y), classes, skipped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--images-per-class", type=int, default=200, help="Maximum images to use from each class. Use 0 for all images.")
    parser.add_argument("--exclude", nargs="*", default=[], help="Class folders to exclude. Default: None.")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    X, y, classes, skipped = load_dataset(args.data, args.images_per_class, args.random_state, args.exclude)
    print(f"\nUsable samples: {len(X)}")
    print(f"Skipped samples: {skipped}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True))
    ])

    print("\nTraining SVM...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))

    labels_sorted = sorted(classes)
    cm = confusion_matrix(y_test, pred, labels=labels_sorted)

    fig_w = max(8, len(labels_sorted) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("ISL Static Gesture - SVM Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=90)
    ax.set_yticklabels(labels_sorted)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=180)
    plt.close(fig)

    joblib.dump(model, "models/isl_svm.joblib")
    with open("models/labels.json", "w", encoding="utf-8") as f:
        json.dump(labels_sorted, f, ensure_ascii=False, indent=2)

    with open("results/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Random state: {args.random_state}\n")
        f.write(f"Test size: {args.test_size}\n")
        f.write(f"Usable samples: {len(X)}\n")
        f.write(f"Skipped samples: {skipped}\n")
        f.write(f"Images per class limit: {args.images_per_class}\n")
        f.write(f"Excluded classes: {args.exclude}\n")
        f.write(f"Test accuracy: {acc:.6f}\n\n")
        f.write(classification_report(y_test, pred, zero_division=0))

    print("\nSaved:")
    print("  models/isl_svm.joblib")
    print("  models/labels.json")
    print("  results/classification_report.txt")
    print("  results/confusion_matrix.png")

if __name__ == "__main__":
    main()
