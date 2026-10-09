import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASCADE_PATH = os.path.join(BASE_DIR, "models", "cascades", "haarcascade_frontalface_default.xml")

face_cascade = None
if os.path.exists(CASCADE_PATH):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
else:
    # Fallback to default cv2 path if available
    cv2_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if os.path.exists(cv2_cascade_path):
        face_cascade = cv2.CascadeClassifier(cv2_cascade_path)


def preprocess_frame_normalization(img_rgb: np.ndarray, apply_noise_reduction: bool = False) -> np.ndarray:
    """
    Preprocessing Layer: Frame Normalization & Color Space Preparation.
    Preserves fine micro-textures and sensor grain for accurate ViT and frequency forensics.
    """
    if img_rgb is None or img_rgb.size == 0:
        return img_rgb

    processed = img_rgb.copy()
    if apply_noise_reduction:
        processed = cv2.bilateralFilter(processed, d=3, sigmaColor=15, sigmaSpace=15)

    return processed


def extract_face_crop(img_rgb: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Extracts a size-invariant padded face crop from a still image.
    Uses multi-scale face localization with 35% bounding box padding.
    """
    if img_rgb is None or img_rgb.size == 0:
        return None

    faces = ()
    if face_cascade is not None and not face_cascade.empty():
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))

    if len(faces) == 0:
        # If no frontal face detected, return centered square crop
        h, w, _ = img_rgb.shape
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        half = min_dim // 2
        crop = img_rgb[max(0, cy-half):min(h, cy+half), max(0, cx-half):min(w, cx+half)]
        if target_size is not None and crop.size > 0:
            crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
        return crop if crop.size > 0 else img_rgb

    # Select largest detected face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    img_h, img_w, _ = img_rgb.shape

    pad = 0.35
    x1 = max(0, int(x - w * pad))
    y1 = max(0, int(y - h * pad))
    x2 = min(img_w, int(x + w * (1.0 + pad)))
    y2 = min(img_h, int(y + h * (1.0 + pad)))

    crop = img_rgb[y1:y2, x1:x2]
    if crop.shape[0] < 30 or crop.shape[1] < 30:
        return None

    if target_size is not None:
        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)

    return crop


def extract_faces_from_video(video_path: str, max_frames: int = 35, target_size: tuple = (224, 224)) -> list:
    """
    Multi-Face Detection & Size-Invariant Tracking across video sequences (MINTIME concept).
    Extracts high-confidence size-standardized face crops and filters out non-face false positives.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    faces = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Adaptive stride to thoroughly scan the video for clean facial appearances
    stride = max(1, total_frames // (max_frames * 2))
    current_frame_idx = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or extracted_count >= max_frames:
            break

        if current_frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape

            detected = ()
            if face_cascade is not None and not face_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # minNeighbors=6 eliminates false positive detections on clothes, furniture and walls
                detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))

            if len(detected) > 0:
                sorted_faces = sorted(detected, key=lambda f: f[2] * f[3], reverse=True)
                for x, y, fw, fh in sorted_faces[:2]:
                    pad = 0.25
                    x1 = max(0, int(x - fw * pad))
                    y1 = max(0, int(y - fh * pad))
                    x2 = min(w, int(x + fw * (1.0 + pad)))
                    y2 = min(h, int(y + fh * (1.0 + pad)))

                    crop = frame_rgb[y1:y2, x1:x2]
                    if crop.shape[0] >= 40 and crop.shape[1] >= 40:
                        if target_size is not None:
                            crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)
                        faces.append(crop)
                        extracted_count += 1
                        if extracted_count >= max_frames:
                            break

        current_frame_idx += 1

    cap.release()
    return faces

