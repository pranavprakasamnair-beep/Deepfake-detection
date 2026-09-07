# 4. SPATIO-TEMPORAL TRANSFORMER & SEQUENCE MODELING MODULE (MINTIME)
temporal_code = '''import cv2
import numpy as np
import torch

def calculate_temporal_motion_glitch(face_crops: list) -> dict:
    """
    Computes inter-frame temporal consistency and motion glitch dynamics.
    Real videos: coherent motion and gradual facial expression transitions.
    Deepfakes: frame-to-frame warping artifacts, blending flickers, and jitter.
    """
    if len(face_crops) < 2:
        return {"temporal_glitch_score": 0.0, "mean_diff": 0.0, "std_diff": 0.0}

    diffs = []
    for i in range(1, len(face_crops)):
        prev = face_crops[i-1].astype(np.float32)
        curr = face_crops[i].astype(np.float32)
        h = min(prev.shape[0], curr.shape[0])
        w = min(prev.shape[1], curr.shape[1])
        prev_res = cv2.resize(prev, (w, h))
        curr_res = cv2.resize(curr, (w, h))
        diff = np.mean(np.abs(curr_res - prev_res)) / 255.0
        diffs.append(diff)

    mean_d = float(np.mean(diffs)) if diffs else 0.0
    std_d = float(np.std(diffs)) if diffs else 0.0

    # High standard deviation of inter-frame difference indicates deepfake frame jitter
    glitch_prob = float(np.clip(std_d / 0.05, 0.0, 1.0))

    return {
        "temporal_glitch_score": round(glitch_prob, 4),
        "mean_diff": round(mean_d, 4),
        "std_diff": round(std_d, 4)
    }

def analyze_vit_temporal_sequence(vit_probs: list) -> dict:
    """
    Analyzes sequence-level Vision Transformer probabilities across time.
    Calculates sequence variance, peak anomaly spikes, and locates suspicious frames.
    """
    if not vit_probs:
        return {"seq_mean_prob": 0.5, "seq_variance": 0.0, "suspicious_frames": []}

    probs_arr = np.array(vit_probs, dtype=np.float32)
    mean_prob = float(np.mean(probs_arr))
    median_prob = float(np.median(probs_arr))
    variance = float(np.var(probs_arr))

    # Suspicious frames: individual frames where ViT probability exceeds manipulation threshold (0.65)
    suspicious_indices = [int(idx) for idx, prob in enumerate(vit_probs) if prob > 0.65]

    return {
        "seq_mean_prob": round(mean_prob, 4),
        "seq_median_prob": round(median_prob, 4),
        "seq_variance": round(variance, 4),
        "suspicious_frames": suspicious_indices
    }

def spatio_temporal_video_analysis(face_crops: list, vit_probs: list) -> dict:
    """
    Consolidated Spatio-Temporal Modeling Module (MINTIME Framework - Section III).
    Combines pixel-level motion glitch analysis with Vision Transformer sequence representations.
    """
    motion_metrics = calculate_temporal_motion_glitch(face_crops)
    seq_metrics = analyze_vit_temporal_sequence(vit_probs)

    # Dynamic temporal fusion
    combined_temporal = (0.65 * motion_metrics["temporal_glitch_score"]) + (0.35 * seq_metrics["seq_mean_prob"])

    return {
        "temporal_score": round(float(np.clip(combined_temporal, 0.0, 1.0)), 4),
        "motion_glitch": motion_metrics["temporal_glitch_score"],
        "seq_mean_prob": seq_metrics["seq_mean_prob"],
        "seq_variance": seq_metrics["seq_variance"],
        "suspicious_frames": seq_metrics["suspicious_frames"]
    }
'''

with open("ml_models/spatio_temporal_expert.py", "w", encoding="utf-8") as f:
    f.write(temporal_code)
print("Wrote ml_models/spatio_temporal_expert.py")
