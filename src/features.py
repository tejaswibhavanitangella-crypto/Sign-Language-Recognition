import cv2
import numpy as np
import mediapipe as mp

_mp_hands = mp.solutions.hands

def create_hands(static_image_mode=True):
    return _mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def extract_features(image_bgr, hands):
    """Return a 63-value normalized landmark vector, or None if no hand is found."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if not result.multi_hand_landmarks:
        return None

    landmarks = result.multi_hand_landmarks[0].landmark
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

    # Make coordinates translation-invariant using the wrist (landmark 0).
    points = points - points[0]

    # Make scale approximately invariant using the maximum XY distance from wrist.
    scale = np.max(np.linalg.norm(points[:, :2], axis=1))
    if scale < 1e-6:
        return None
    points = points / scale

    return points.reshape(-1).astype(np.float32)

def draw_landmarks(image_bgr, hands, text=None):
    """Draw detected landmarks for the webcam demo."""
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                _mp_hands.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )
    if text:
        cv2.putText(
            image_bgr, text, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
        )
    return image_bgr
