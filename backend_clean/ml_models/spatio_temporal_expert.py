import cv2
import numpy as np
import torch

def calculate_temporal_motion_glitch(face_crops: list) -> dict:
    """
    Computes inter-frame temporal consistency and motion glitch dynamics.
    Real videos: coherent motion and gradual facial expression transitions,
    but CAN include fast natural motion (hand gestures, head turns, shaky
    handheld footage) which also produces large raw pixel differences.
    Deepfakes: frame-to-frame warping artifacts, blending flickers, jitter
    that is NOT explained by overall motion magnitude alone.
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

    # NOTE (recalibrated): raw std of inter-frame diff alone can't distinguish
    # deepfake jitter from ordinary fast handheld motion (hand gestures, quick
    # head turns) -- both produce high std_d. Widening the threshold band so
    # only unusually spiky/erratic jitter relative to the overall motion level
    # triggers a high score, rather than any fast-but-coherent motion.
    #
    # Additionally normalize glitch relative to the mean motion level: a video
    # with consistently high motion (mean_d) but proportionally low variance
    # (std_d) is coherent fast motion, not glitching. A video with jitter that
    # spikes far above its own average motion is more likely synthetic.
    if mean_d > 1e-4:
        relative_spikiness = std_d / (mean_d + 1e-4)
    else:
        relative_spikiness = 0.0

    # Relative spikiness distinguishes sudden unnatural warping flickers from smooth natural gestures
    if mean_d > 0.02:
        relative_spikiness = std_d / (mean_d + 1e-4)
    else:
        relative_spikiness = 0.0

    raw_glitch = float(np.clip((std_d - 0.06) / 0.08, 0.0, 1.0))
    spikiness_gate = float(np.clip((relative_spikiness - 1.1) / 0.8, 0.0, 1.0))
    glitch_prob = float(raw_glitch * 0.5 + spikiness_gate * 0.5)

    return {
        "temporal_glitch_score": round(glitch_prob, 4),
        "mean_diff": round(mean_d, 4),
        "std_diff": round(std_d, 4)
    }

def analyze_vit_temporal_sequence(vit_probs: list) -> dict:
    """
    Analyzes sequence-level Vision Transformer probabilities across time.
    Uses median temporal filtering to prevent momentary hand occlusions, blinks,
    and fast head turns from distorting the overall sequence score.
    """
    if not vit_probs:
        return {"seq_mean_prob": 0.5, "seq_median_prob": 0.5, "effective_vit_prob": 0.5, "seq_variance": 0.0, "suspicious_frames": []}

    probs_arr = np.array(vit_probs, dtype=np.float32)
    mean_prob = float(np.mean(probs_arr))
    median_prob = float(np.median(probs_arr))
    variance = float(np.var(probs_arr))

    k = max(1, len(probs_arr) // 4)
    sorted_probs = np.sort(probs_arr)[::-1]
    peak_prob = float(np.mean(sorted_probs[:k]))

    high_fake_count = int(np.sum(probs_arr > 0.75))
    high_fake_ratio = high_fake_count / len(probs_arr)

    # Robust temporal sequence aggregation:
    # If multiple frames exhibit undeniable deepfake manipulation (>=3 frames with >0.75 or >20% frames with >0.75):
    if (high_fake_count >= 3 or high_fake_ratio >= 0.20) and peak_prob > 0.80:
        effective_vit_prob = float(0.70 * peak_prob + 0.30 * mean_prob)
    elif median_prob < 0.30:
        effective_vit_prob = float(0.80 * median_prob + 0.20 * mean_prob)
    else:
        effective_vit_prob = float(0.60 * peak_prob + 0.40 * median_prob)

    suspicious_indices = [int(idx) for idx, prob in enumerate(vit_probs) if prob > 0.60]

    return {
        "seq_mean_prob": round(mean_prob, 4),
        "seq_median_prob": round(median_prob, 4),
        "effective_vit_prob": round(effective_vit_prob, 4),
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

    vit_signal = seq_metrics.get("effective_vit_prob", seq_metrics["seq_median_prob"])
    motion_signal = motion_metrics["temporal_glitch_score"]

    # When the median facial frames verify authentic human features (vit_signal < 0.45),
    # the temporal stream confirms authentic human motion regardless of camera shake.
    if vit_signal < 0.45:
        combined_temporal = float(vit_signal * 0.85)
    elif vit_signal > 0.55:
        combined_temporal = float(max(vit_signal, 0.70 * vit_signal + 0.30 * motion_signal))
    else:
        combined_temporal = float(0.70 * vit_signal + 0.30 * motion_signal)

    return {
        "temporal_score": round(float(np.clip(combined_temporal, 0.0, 1.0)), 4),
        "motion_glitch": motion_metrics["temporal_glitch_score"],
        "seq_mean_prob": seq_metrics["seq_mean_prob"],
        "effective_vit_prob": seq_metrics.get("effective_vit_prob", seq_metrics["seq_median_prob"]),
        "seq_variance": seq_metrics["seq_variance"],
        "suspicious_frames": seq_metrics["suspicious_frames"]
    }