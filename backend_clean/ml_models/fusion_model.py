import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sigmoid_engagement(value: float, threshold: float, slope: float = 25.0) -> float:
    """
    Smoothly maps a raw branch score to a 0-1 signal, centered on `threshold`.
    """
    exp_arg = np.clip(-slope * (value - threshold), -50.0, 50.0)
    return float(1.0 / (1.0 + np.exp(exp_arg)))


def _compute_verdict(score: float, uncertain_low: float = 0.40, uncertain_high: float = 0.60) -> tuple[str, float, bool]:
    """
    Converts a continuous fake-probability score into a REAL / FAKE / UNCERTAIN
    verdict plus a 0-1 confidence value.
    """
    if score < uncertain_low:
        verdict = "REAL"
        confidence = float(np.clip(1.0 - score, 0.50, 1.0))
    elif score > uncertain_high:
        verdict = "FAKE"
        confidence = float(np.clip(score, 0.50, 1.0))
    else:
        verdict = "UNCERTAIN"
        half_band = (uncertain_high - uncertain_low) / 2.0
        confidence = float(np.clip(1.0 - abs(score - 0.5) / half_band, 0.0, 1.0)) if half_band > 0 else 0.0

    return verdict, round(confidence, 4), verdict == "UNCERTAIN"


# ---------------------------------------------------------------------------
# Image fusion
# ---------------------------------------------------------------------------

def apply_rule_based_fallback(
    preliminary_score: float,
    spatial_data: dict,
    frequency_data: dict,
    attention_data: dict,
    vit_prob: float
) -> tuple[float, list[str]]:
    """
    Fallback and Validation Mechanism (Survey Paper Section III).
    Applies forensic rules ONLY when multiple independent cues corroborate forgery.
    Never inflates clean authentic images.
    """
    fallback_flags = []

    bound_score = spatial_data.get("boundary_score", 0.0)
    fft_score = frequency_data.get("fft_score", 0.0)
    noise_std = frequency_data.get("noise_std", 2.5)

    if bound_score > 0.65:
        fallback_flags.append("Elevated boundary seam distortion detected")
    if fft_score > 0.75:
        fallback_flags.append("Abnormal high-frequency power spectrum drop-off")

    # Only apply positive boost if multiple strong forensic flags trigger AND ViT is not already firmly real
    if len(fallback_flags) >= 2 and vit_prob > 0.40:
        adjusted_score = max(preliminary_score, 0.75)
    else:
        adjusted_score = preliminary_score

    return float(np.clip(adjusted_score, 0.0, 1.0)), fallback_flags


def fuse_image_multi_branch(
    vit_prob: float,
    spatial_data: dict,
    frequency_data: dict,
    attention_data: dict
) -> dict:
    """
    Comprehensive Multi-Branch Image Fusion (Section III).
    Branches:
    1. Vision Transformer (ViT) representation (0.50)
    2. Frequency Domain (FFT + DCT + Noise) (0.20)
    3. CNN Spatial Features (Texture + Seams + Lighting) (0.18)
    4. Attention-Enhanced Critical Regions (0.12)
    """
    spatial_score = spatial_data.get("spatial_fake_score", 0.5)
    freq_score = frequency_data.get("frequency_fake_score", 0.5)
    attn_score = attention_data.get("attention_anomaly_score", 0.5)

    base_fused = (
        0.50 * vit_prob +
        0.20 * freq_score +
        0.18 * spatial_score +
        0.12 * attn_score
    )

    final_score, fallback_notes = apply_rule_based_fallback(
        base_fused, spatial_data, frequency_data, attention_data, vit_prob
    )

    # Multi-branch Decision Calibration:
    # 1. Strong deepfake detection in ViT or multiple forensic streams -> FAKE
    if vit_prob > 0.55 or spatial_score > 0.60 or freq_score > 0.60:
        max_expert = max(vit_prob, freq_score, spatial_score)
        final_score = max(final_score, max_expert * 0.95, 0.90)
    # 2. Authentic human facial features -> REAL
    elif vit_prob < 0.45 and spatial_score < 0.45 and freq_score < 0.45:
        final_score = min(final_score, 0.10)
    elif vit_prob < 0.40:
        final_score = min(final_score, 0.12)

    verdict, confidence, needs_review = _compute_verdict(final_score)

    return {
        "final_score": round(final_score, 4),
        "verdict": verdict,
        "confidence": confidence,
        "needs_review": needs_review,
        "vit_prob": round(vit_prob, 4),
        "spatial_score": round(spatial_score, 4),
        "frequency_score": round(freq_score, 4),
        "attention_score": round(attn_score, 4),
        "fallback_triggered": len(fallback_notes) > 0,
        "fallback_notes": fallback_notes
    }


# ---------------------------------------------------------------------------
# Video fusion
# ---------------------------------------------------------------------------

def fuse_video_multi_branch(
    temporal_data: dict,
    spatial_data: dict,
    frequency_data: dict,
    attention_data: dict,
    audio_score: float = None
) -> dict:
    """
    Comprehensive Multi-Branch Video Fusion (Section III & MINTIME).
    Branches:
    1. Spatio-Temporal Inter-frame Glitch & ViT Dynamics (0.55)
    2. Frequency Domain (FFT + DCT + Noise) (0.18)
    3. Spatial Artifacts & Seams (0.15)
    4. Attention-Enhanced Anomaly (0.12)
    + Optional Multimodal Audio Integration (30% audio, 70% video if audio exists)
    """
    temp_score = temporal_data.get("temporal_score", 0.5)
    spatial_score = spatial_data.get("spatial_fake_score", 0.5)
    freq_score = frequency_data.get("frequency_fake_score", 0.5)
    attn_score = attention_data.get("attention_anomaly_score", 0.5)

    video_visual_score = (
        0.55 * temp_score +
        0.18 * freq_score +
        0.15 * spatial_score +
        0.12 * attn_score
    )

    escalation_notes = []
    if temp_score > 0.55:
        escalation_notes.append("Temporal inter-frame jitter or facial synthesis detected")
    if spatial_score > 0.60:
        escalation_notes.append("Spatial boundary seam or texture anomaly detected")
    if freq_score > 0.60:
        escalation_notes.append("High-frequency power spectrum distortion detected")

    # Video Decision Matrix:
    # 1. Consistent Deepfake Identity / Synthesis Glitch -> FAKE
    if temp_score > 0.55 or spatial_score > 0.60 or freq_score > 0.60:
        max_expert = max(temp_score, spatial_score, freq_score)
        video_visual_score = max(video_visual_score, max_expert * 0.95, 0.90)
    # 2. Genuine Human Motion & Facial Dynamics -> REAL
    elif temp_score < 0.40:
        video_visual_score = min(video_visual_score, 0.10)

    if audio_score is not None:
        overall_score = (0.70 * video_visual_score) + (0.30 * audio_score)
    else:
        overall_score = video_visual_score

    overall_score = float(np.clip(overall_score, 0.0, 1.0))

    verdict, confidence, needs_review = _compute_verdict(overall_score)

    return {
        "final_score": round(overall_score, 4),
        "verdict": verdict,
        "confidence": confidence,
        "needs_review": needs_review,
        "visual_score": round(float(np.clip(video_visual_score, 0.0, 1.0)), 4),
        "temporal_score": round(temp_score, 4),
        "spatial_score": round(spatial_score, 4),
        "frequency_score": round(freq_score, 4),
        "attention_score": round(attn_score, 4),
        "audio_score": round(audio_score, 4) if audio_score is not None else None,
        "escalation_notes": escalation_notes,
        "suspicious_frames": temporal_data.get("suspicious_frames", [])
    }