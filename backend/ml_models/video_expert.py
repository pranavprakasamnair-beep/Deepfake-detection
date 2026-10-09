"""
Video Expert — Frame-by-Frame Face Tracking (MINTIME) + Temporal Consistency Analysis.
"""
import logging
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_face_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def extract_video_faces(video_path: str, max_frames: int = 30) -> List[Image.Image]:
    """
    Extracts high-quality face crops across video sequence using adaptive stride.
    minNeighbors=6 ensures non-face clothing/wall patterns are rejected.
    """
    cascade = _get_cascade()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 60

    # Adapt stride so we scan throughout the video
    stride = max(1, total_frames // max_frames)
    faces = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or len(faces) >= max_frames:
            break

        if frame_idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(60, 60))
            if len(detected) > 0:
                # Select the largest human face
                x, y, fw, fh = max(detected, key=lambda f: f[2] * f[3])
                h, w, _ = frame.shape
                pad = 0.20
                x1 = max(0, int(x - fw * pad))
                y1 = max(0, int(y - fh * pad))
                x2 = min(w, int(x + fw * (1.0 + pad)))
                y2 = min(h, int(y + fh * (1.0 + pad)))
                crop = frame[y1:y2, x1:x2]
                if crop.shape[0] >= 40 and crop.shape[1] >= 40:
                    faces.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))

        frame_idx += 1

    cap.release()
    return faces


def measure_temporal_glitch(faces: List[Image.Image]) -> float:
    """Detects frame-to-frame pixel jumps characteristic of AI video reenactment."""
    if len(faces) < 3:
        return 0.20

    diffs = []
    prev = np.array(faces[0].convert("L").resize((96, 96)), dtype=np.float32)

    for f in faces[1:]:
        curr = np.array(f.convert("L").resize((96, 96)), dtype=np.float32)
        diff = float(np.mean(np.abs(curr - prev)))
        diffs.append(diff)
        prev = curr

    diffs_arr = np.array(diffs)
    mean_d = float(np.mean(diffs_arr))
    std_d = float(np.std(diffs_arr))
    spikiness = std_d / (mean_d + 1e-6)

    # Abnormal spikiness indicates deepfake frame jitters
    return float(np.clip(spikiness / 1.8, 0.0, 1.0))


def analyze_video(video_path: str, registry) -> dict:
    """Analyze video with multi-frame face tracking and temporal consistency."""
    from ml_models.image_expert import _predict_face_prob

    faces = extract_video_faces(video_path, max_frames=30)
    if not faces:
        logger.warning(f"No clean faces detected in {video_path}")
        return {
            "score": 0.50,
            "verdict": "UNKNOWN",
            "details": {"note": "No clear human faces detected in video"},
        }

    frame_probs = [_predict_face_prob(f, registry) for f in faces]
    probs_arr = np.array(frame_probs, dtype=np.float32)

    median_prob = float(np.median(probs_arr))
    mean_prob = float(np.mean(probs_arr))
    k = max(1, len(probs_arr) // 3)
    peak_prob = float(np.mean(np.sort(probs_arr)[::-1][:k]))

    high_fakes = int(np.sum(probs_arr > 0.65))
    high_fake_ratio = high_fakes / len(probs_arr)

    temporal_glitch = measure_temporal_glitch(faces)

    # Sequence aggregation:
    # If >= 2 frames or >= 20% of frames show deepfake manipulation, and peak is strong:
    if (high_fakes >= 2 or high_fake_ratio >= 0.20) and peak_prob > 0.70:
        video_score = 0.88 * peak_prob + 0.12 * mean_prob
    elif median_prob < 0.20:
        video_score = 0.90 * median_prob + 0.10 * mean_prob
    else:
        video_score = 0.60 * peak_prob + 0.40 * median_prob

    # Boost slightly if temporal glitch is pronounced
    if temporal_glitch > 0.60:
        video_score = min(1.0, video_score + 0.10)

    verdict = "FAKE" if video_score > 0.50 else "REAL"

    return {
        "score": round(float(np.clip(video_score, 0.0, 1.0)), 4),
        "verdict": verdict,
        "details": {
            "frames_analyzed": len(faces),
            "median_fake_prob": round(median_prob, 4),
            "peak_fake_prob": round(peak_prob, 4),
            "high_fake_frames": high_fakes,
            "temporal_glitch_score": round(temporal_glitch, 4),
        }
    }
