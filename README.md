# ISL Static Alphabet & Number Recognition using SVM

A traditional machine-learning project for recognizing isolated/static Indian Sign Language (ISL) hand gestures from images or a webcam.

## Scope
- Primary classifier: Support Vector Machine (SVM)
- Hand features: MediaPipe Hands 21 landmarks (x, y, z)
- Image processing: OpenCV
- No CNN / TensorFlow / Keras / deep-learning classifier
- Dataset: place the downloaded ISL image dataset under `data/dataset/`
- Training subset: automatically uses a maximum of **200 images per class** (configurable)
- Default static-only training excludes **J and Z**
- The code supports folder-per-class datasets such as:
  data/dataset/A/*.jpg
  data/dataset/B/*.jpg
  ...
  data/dataset/1/*.jpg

### Static-only note
Some ISL alphabet signs, notably J and Z, involve movement. If your guide requires strictly static gestures, remove those class folders before training. The software itself does not decide whether a sign is static; it trains on the folders you provide.

## 1. Install Python
Recommended: Python 3.10 or 3.11.

## 2. Open this project in VS Code
Extract the ZIP and open the `ISL_Static_Gesture_SVM` folder.

## 3. Create a virtual environment

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install packages
```bash
pip install -r requirements.txt
```

If your computer uses `python3`/`pip3`, use those commands instead.

## 5. Download the dataset
Use the Kaggle ISL dataset discussed with your project guide. Put the class folders under:

```text
data/
└── dataset/
    ├── A/
    ├── B/
    ├── C/
    ├── ...
    ├── Z/
    ├── 1/
    ├── ...
    └── 9/
```

The training script recursively reads image files from these class folders.

## 6. Train the SVM
From the project root:

```bash
python src/train.py
```

By default, the program automatically selects **up to 200 images per class**. You do not need to copy or delete images manually. With 35 classes, this limits processing to about 7,000 images before hand-detection failures are removed. For the strict static-only project, J and Z are excluded by default.

Optional arguments:
```bash
python src/train.py --data data/dataset --test-size 0.2 --images-per-class 200
```

Use all images from each class:
```bash
python src/train.py --images-per-class 0
```

Include J and Z if your guide requires them:
```bash
python src/train.py --exclude
```

The script:
1. Finds all class folders.
2. Selects up to 200 images from each class automatically.
3. Excludes J and Z by default for static-only ISL.
4. Reads the selected images and labels.
5. Detects one hand with MediaPipe.
6. Extracts 21 landmarks.
7. Converts them into normalized 63-value feature vectors.
8. Splits the data into train/test sets.
9. Trains an RBF-kernel SVM.
10. Prints a classification report.
11. Saves the model to `models/isl_svm.joblib`.
12. Saves the class names to `models/labels.json`.
13. Saves a confusion matrix to `results/confusion_matrix.png`.

Images where a hand cannot be detected are skipped and counted.

## 7. Run evaluation on the saved model
```bash
python src/evaluate.py
```

This retrains no model. It evaluates using a newly generated test split from the dataset. For a formal report, use the metrics produced by the training run and record the exact split/seed shown by the program.

## 8. Run the webcam demo
After successful training:

```bash
python src/realtime.py
```

A webcam window will open.

- Show one supported static hand gesture.
- Hold it steady.
- The predicted class is displayed on screen.
- Press `q` to quit.

You can also specify a camera:
```bash
python src/realtime.py --camera 0
```

## Important troubleshooting

### MediaPipe import error
Try:
```bash
pip install --upgrade mediapipe
```
and verify your Python version is supported by the installed MediaPipe release.

### Camera does not open
- Close other apps using the camera.
- Try `--camera 1`.
- Check Windows camera permissions.

### Many images are skipped
This means MediaPipe could not find a hand in those images. Check image quality and whether the dataset contains clear, single-hand images.

### Prediction is poor
Do not change the algorithm immediately. First inspect:
- class labels
- duplicate/near-duplicate images
- lighting/background variation
- incorrect images
- hand detection failures
- train/test leakage

For a stronger real-world evaluation, test the final model using photographs/webcam frames from people who are not represented in the training images.

## Project workflow

```text
ISL Dataset
    ↓
OpenCV image loading
    ↓
MediaPipe hand detection
    ↓
21 hand landmarks
    ↓
Wrist-relative + scale normalization
    ↓
63 numeric features
    ↓
RBF SVM
    ↓
ISL alphabet/number prediction
```

## Files

- `src/features.py` — landmark extraction and normalization
- `src/train.py` — dataset loading and SVM training
- `src/evaluate.py` — model evaluation
- `src/realtime.py` — webcam prediction
- `models/` — saved trained model and labels
- `results/` — evaluation output
